"""
Map/Positioning Analyzer for Valorant VODs.

Analyzes player positioning on the minimap to detect:
- Whether the player is in common/good positions
- Rotation patterns and timing
- Exposed positioning (too far forward/back)
- Site anchor vs rotator behavior

This is an additional analysis dimension for coaching.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MapFrame:
    timestamp: float
    minimap_detected: bool
    player_position: tuple[int, int]  # (x, y) on minimap
    zone: str  # "a_site", "b_site", "mid", "spawn", "unknown"
    is_exposed: bool  # too far forward without team support
    is_rotating: bool  # player is moving between zones
    zone_density: float  # how clustered teammates are


@dataclass
class MapAnalysisResult:
    positioning_score: float = 0.0
    time_in_zones: dict = field(default_factory=dict)  # zone -> percentage
    rotation_count: int = 0
    avg_rotation_time: float = 0.0
    exposed_positioning_pct: float = 0.0
    zone_timeline: list[dict] = field(default_factory=list)
    positioning_events: list[dict] = field(default_factory=list)
    score: float = 0.0


# ── Map-specific callout definitions ──────────────────────────────────
# Each map defines rectangular zones on a normalised [0,1]x[0,1] minimap.
# Format: list of (name, x_min, y_min, x_max, y_max).
# The first match wins, so put more specific zones before generic ones.

_BIND_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Short",   0.00, 0.00, 0.20, 0.20),
    ("A Site",    0.00, 0.20, 0.25, 0.45),
    ("A Bath",    0.00, 0.45, 0.20, 0.60),
    ("A Lobby",   0.00, 0.60, 0.25, 0.80),
    ("B Short",   0.75, 0.00, 1.00, 0.20),
    ("B Site",    0.75, 0.20, 1.00, 0.45),
    ("B Long",    0.75, 0.45, 1.00, 0.65),
    ("B Lobby",   0.75, 0.65, 1.00, 0.85),
    ("Mid",       0.30, 0.20, 0.70, 0.55),
    ("CT Spawn",  0.30, 0.00, 0.70, 0.20),
    ("T Spawn",   0.30, 0.80, 0.70, 1.00),
]

_HAVEN_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Short",   0.00, 0.00, 0.20, 0.25),
    ("A Long",    0.00, 0.25, 0.15, 0.50),
    ("A Site",    0.00, 0.10, 0.25, 0.35),
    ("A Lobby",   0.00, 0.50, 0.20, 0.75),
    ("C Long",    0.80, 0.00, 1.00, 0.30),
    ("C Site",    0.75, 0.10, 1.00, 0.40),
    ("C Lobby",   0.80, 0.40, 1.00, 0.65),
    ("B Site",    0.35, 0.05, 0.65, 0.30),
    ("Mid Window",0.30, 0.30, 0.50, 0.50),
    ("Mid",       0.25, 0.30, 0.75, 0.55),
    ("Garage",    0.50, 0.50, 0.75, 0.70),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.10),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_ASCENT_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Short",   0.15, 0.15, 0.35, 0.35),
    ("A Lobby",   0.00, 0.55, 0.20, 0.75),
    ("B Main",    0.80, 0.35, 1.00, 0.55),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Lobby",   0.80, 0.55, 1.00, 0.75),
    ("Mid Top",   0.35, 0.15, 0.65, 0.35),
    ("Mid",       0.30, 0.35, 0.70, 0.55),
    ("Mid Bottom",0.35, 0.55, 0.65, 0.70),
    ("Market",    0.55, 0.20, 0.70, 0.40),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_SPLIT_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Ramp",    0.15, 0.20, 0.30, 0.40),
    ("A Lobby",   0.00, 0.55, 0.25, 0.75),
    ("B Main",    0.80, 0.35, 1.00, 0.55),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Lobby",   0.75, 0.55, 1.00, 0.75),
    ("Mid",       0.30, 0.25, 0.70, 0.55),
    ("Vent",      0.40, 0.15, 0.60, 0.25),
    ("Sewer",     0.40, 0.55, 0.60, 0.70),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_ICEBOX_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Belt",    0.10, 0.35, 0.30, 0.50),
    ("A Main",    0.00, 0.50, 0.25, 0.70),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Orange",  0.70, 0.35, 0.90, 0.50),
    ("B Main",    0.75, 0.50, 1.00, 0.70),
    ("Mid",       0.30, 0.25, 0.70, 0.55),
    ("Kitchen",   0.50, 0.15, 0.70, 0.30),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_BREEZE_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.05, 0.30, 0.30),
    ("A Main",    0.00, 0.30, 0.20, 0.55),
    ("A Hall",    0.20, 0.15, 0.35, 0.35),
    ("A Lobby",   0.00, 0.55, 0.20, 0.75),
    ("B Site",    0.70, 0.05, 1.00, 0.30),
    ("B Main",    0.80, 0.30, 1.00, 0.55),
    ("B Lobby",   0.75, 0.55, 1.00, 0.75),
    ("Mid",       0.30, 0.25, 0.70, 0.50),
    ("Mid Nest",  0.40, 0.10, 0.60, 0.25),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.10),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_FRACTURE_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Rope",    0.15, 0.25, 0.30, 0.40),
    ("A Lobby",   0.00, 0.55, 0.20, 0.75),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Main",    0.80, 0.35, 1.00, 0.55),
    ("B Arcade",  0.75, 0.25, 0.90, 0.40),
    ("B Lobby",   0.75, 0.55, 1.00, 0.75),
    ("Mid",       0.30, 0.25, 0.70, 0.55),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_PEARL_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Art",     0.15, 0.20, 0.30, 0.40),
    ("A Lobby",   0.00, 0.55, 0.25, 0.75),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Main",    0.80, 0.35, 1.00, 0.55),
    ("B Hall",    0.75, 0.20, 0.90, 0.40),
    ("B Lobby",   0.75, 0.55, 1.00, 0.75),
    ("Mid Top",   0.35, 0.15, 0.65, 0.35),
    ("Mid",       0.30, 0.35, 0.70, 0.55),
    ("Mid Bottom",0.35, 0.55, 0.65, 0.70),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_LOTUS_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.10, 0.25, 0.35),
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Root",    0.10, 0.20, 0.25, 0.40),
    ("A Lobby",   0.00, 0.55, 0.20, 0.75),
    ("C Site",    0.75, 0.10, 1.00, 0.35),
    ("C Main",    0.80, 0.35, 1.00, 0.55),
    ("C Lobby",   0.75, 0.55, 1.00, 0.75),
    ("B Site",    0.35, 0.05, 0.65, 0.25),
    ("B Main",    0.35, 0.25, 0.50, 0.45),
    ("Mid",       0.25, 0.35, 0.75, 0.55),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.10),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

_SUNSET_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",    0.00, 0.10, 0.30, 0.35),
    ("A Main",    0.00, 0.35, 0.20, 0.55),
    ("A Lobby",   0.00, 0.55, 0.20, 0.75),
    ("B Site",    0.70, 0.10, 1.00, 0.35),
    ("B Main",    0.80, 0.35, 1.00, 0.55),
    ("B Lobby",   0.75, 0.55, 1.00, 0.75),
    ("Mid Top",   0.35, 0.15, 0.65, 0.35),
    ("Mid",       0.30, 0.35, 0.70, 0.55),
    ("Mid Bottom",0.35, 0.55, 0.65, 0.70),
    ("CT Spawn",  0.35, 0.00, 0.65, 0.15),
    ("T Spawn",   0.35, 0.80, 0.65, 1.00),
]

# Fallback generic callouts (used when map is unknown)
_GENERIC_CALLOUTS: list[tuple[str, float, float, float, float]] = [
    ("A Site",  0.00, 0.00, 0.30, 0.25),
    ("A Main",  0.00, 0.25, 0.25, 0.50),
    ("A Lobby", 0.00, 0.50, 0.25, 0.75),
    ("B Site",  0.70, 0.00, 1.00, 0.25),
    ("B Main",  0.70, 0.25, 1.00, 0.50),
    ("B Lobby", 0.70, 0.50, 1.00, 0.75),
    ("Mid",     0.25, 0.20, 0.70, 0.55),
    ("CT Spawn",0.30, 0.00, 0.70, 0.15),
    ("T Spawn", 0.30, 0.80, 0.70, 1.00),
]

MAP_CALLOUTS: dict[str, list[tuple[str, float, float, float, float]]] = {
    "bind":     _BIND_CALLOUTS,
    "haven":    _HAVEN_CALLOUTS,
    "ascent":   _ASCENT_CALLOUTS,
    "split":    _SPLIT_CALLOUTS,
    "icebox":   _ICEBOX_CALLOUTS,
    "breeze":   _BREEZE_CALLOUTS,
    "fracture":  _FRACTURE_CALLOUTS,
    "pearl":    _PEARL_CALLOUTS,
    "lotus":    _LOTUS_CALLOUTS,
    "sunset":   _SUNSET_CALLOUTS,
    "generic":  _GENERIC_CALLOUTS,
}


class MapAnalyzer:
    """
    Analyzes player positioning from the Valorant minimap.

    The minimap in Valorant is located in the top-left corner of the screen.
    It shows player positions, teammate positions, and the map layout.

    Supports map-specific callouts for all 10 competitive maps:
    Bind, Haven, Ascent, Split, Icebox, Breeze, Fracture, Pearl, Lotus, Sunset.

    Key metrics:
    - Zone Distribution: time spent in each area of the map
    - Rotation Speed: how fast the player rotates between sites
    - Exposed Positioning: being too far forward or in dangerous spots
    - Site Presence: anchoring vs rotating patterns
    """

    # Minimap is in the top-left corner, roughly 15% of screen width/height
    MINIMAP_X_RATIO = 0.0
    MINIMAP_Y_RATIO = 0.0
    MINIMAP_W_RATIO = 0.14
    MINIMAP_H_RATIO = 0.22

    # Player indicator is typically a bright green/teal arrow on the minimap
    PLAYER_COLOR_LOWER_HSV = np.array([75, 100, 150])
    PLAYER_COLOR_UPPER_HSV = np.array([100, 255, 255])

    # Teammate color (green circles)
    TEAM_COLOR_LOWER_HSV = np.array([55, 80, 120])
    TEAM_COLOR_UPPER_HSV = np.array([85, 255, 255])

    # Enemy color (red)
    ENEMY_COLOR_LOWER_HSV = np.array([0, 120, 120])
    ENEMY_COLOR_UPPER_HSV = np.array([10, 255, 255])

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        map_name: str = "generic",
    ):
        self.width, self.height = resolution
        self.map_name = map_name.lower().strip()
        self.callouts = MAP_CALLOUTS.get(self.map_name, _GENERIC_CALLOUTS)
        self.frames: list[MapFrame] = []
        self.prev_zone: str = "unknown"
        self.zone_changes: list[dict] = []
        self.last_zone_change_time: float = 0.0

    def extract_minimap(self, frame: np.ndarray) -> np.ndarray:
        """Extract the minimap region from the frame."""
        h, w = frame.shape[:2]
        x1 = int(w * self.MINIMAP_X_RATIO)
        y1 = int(h * self.MINIMAP_Y_RATIO)
        x2 = int(w * (self.MINIMAP_X_RATIO + self.MINIMAP_W_RATIO))
        y2 = int(h * (self.MINIMAP_Y_RATIO + self.MINIMAP_H_RATIO))
        return frame[y1:y2, x1:x2]

    def detect_player_position(self, minimap: np.ndarray) -> tuple[bool, tuple[int, int]]:
        """
        Detect the player's position on the minimap.

        The player indicator is a bright colored arrow/triangle.
        We look for the brightest cluster of the player's color.
        """
        if minimap.size == 0:
            return False, (0, 0)

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # Look for player indicator color
        mask = cv2.inRange(hsv, self.PLAYER_COLOR_LOWER_HSV, self.PLAYER_COLOR_UPPER_HSV)

        # Also check for bright white/cyan indicators
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_or(mask, white_mask)

        # Find the largest contour (player indicator)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: use center of minimap as approximate position
            h, w = minimap.shape[:2]
            return False, (w // 2, h // 2)

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return True, (cx, cy)

        return False, (minimap.shape[1] // 2, minimap.shape[0] // 2)

    def classify_zone(
        self, position: tuple[int, int], minimap_size: tuple[int, int]
    ) -> str:
        """
        Classify the player's position into a map callout.

        Uses the map-specific callout rectangles loaded at init time.
        Falls back to a generic zone if no callout matches.
        """
        w, h = minimap_size
        if w == 0 or h == 0:
            return "unknown"

        rx = position[0] / w  # relative x (0-1)
        ry = position[1] / h  # relative y (0-1)

        for name, x1, y1, x2, y2 in self.callouts:
            if x1 <= rx <= x2 and y1 <= ry <= y2:
                return name

        # Fallback generic classification
        if ry > 0.80:
            return "T Spawn"
        if ry < 0.15:
            return "CT Spawn"
        return "Mid"

    def detect_teammates_nearby(self, minimap: np.ndarray, player_pos: tuple[int, int]) -> int:
        """Count teammates near the player on the minimap."""
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.TEAM_COLOR_LOWER_HSV, self.TEAM_COLOR_UPPER_HSV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nearby = 0
        proximity_threshold = max(minimap.shape[:2]) * 0.25

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - player_pos[0]) ** 2 + (cy - player_pos[1]) ** 2)
                if dist < proximity_threshold:
                    nearby += 1

        return nearby

    def assess_exposure(self, zone: str, teammates_nearby: int) -> bool:
        """
        Determine if the player is in an exposed/dangerous position.

        Being in an aggressive zone without teammates nearby = exposed.
        """
        # Normalise zone name for comparison (callout names are title-case)
        z = zone.lower()
        aggressive_keywords = {"site", "main", "long", "short"}
        if any(kw in z for kw in aggressive_keywords) and teammates_nearby < 1:
            return True

        # Mid with no support is risky
        if "mid" in z and teammates_nearby < 1:
            return True

        return False

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
    ) -> MapFrame:
        """Process a single frame for map/positioning analysis."""
        minimap = self.extract_minimap(frame)

        if minimap.size == 0:
            mf = MapFrame(
                timestamp=timestamp,
                minimap_detected=False,
                player_position=(0, 0),
                zone="unknown",
                is_exposed=False,
                is_rotating=False,
                zone_density=0.0,
            )
            self.frames.append(mf)
            return mf

        detected, position = self.detect_player_position(minimap)
        minimap_h, minimap_w = minimap.shape[:2]
        zone = self.classify_zone(position, (minimap_w, minimap_h))

        teammates_nearby = self.detect_teammates_nearby(minimap, position)
        is_exposed = self.assess_exposure(zone, teammates_nearby)

        # Detect rotation (zone change)
        is_rotating = False
        if zone != self.prev_zone and self.prev_zone != "unknown" and zone != "unknown":
            is_rotating = True
            self.zone_changes.append({
                "timestamp": timestamp,
                "from_zone": self.prev_zone,
                "to_zone": zone,
                "duration": timestamp - self.last_zone_change_time if self.last_zone_change_time > 0 else 0,
            })
            self.last_zone_change_time = timestamp

        self.prev_zone = zone

        # Zone density (how many teammates in same area)
        zone_density = min(1.0, teammates_nearby / 4.0)

        mf = MapFrame(
            timestamp=timestamp,
            minimap_detected=detected,
            player_position=position,
            zone=zone,
            is_exposed=is_exposed,
            is_rotating=is_rotating,
            zone_density=zone_density,
        )
        self.frames.append(mf)
        return mf

    def generate_results(self) -> MapAnalysisResult:
        """Generate map/positioning analysis results."""
        if not self.frames:
            return MapAnalysisResult(score=50.0)

        # Time in zones
        zone_counts: dict[str, int] = {}
        for f in self.frames:
            zone_counts[f.zone] = zone_counts.get(f.zone, 0) + 1

        total_frames = len(self.frames)
        time_in_zones = {
            zone: round((count / total_frames) * 100, 1)
            for zone, count in zone_counts.items()
        }

        # Rotation count and average time
        rotation_count = len(self.zone_changes)
        avg_rotation_time = 0.0
        if rotation_count > 1:
            rotation_durations = [
                zc["duration"] for zc in self.zone_changes
                if zc["duration"] > 0
            ]
            avg_rotation_time = float(np.mean(rotation_durations)) if rotation_durations else 0.0

        # Exposed positioning percentage
        exposed_frames = sum(1 for f in self.frames if f.is_exposed)
        exposed_pct = (exposed_frames / total_frames) * 100

        # Zone timeline (sampled)
        zone_timeline = [
            {
                "timestamp": f.timestamp,
                "zone": f.zone,
                "exposed": f.is_exposed,
                "teammates_nearby": f.zone_density,
            }
            for f in self.frames[::5]
        ]

        # Positioning events (rotations, exposed moments)
        positioning_events = []
        for zc in self.zone_changes:
            positioning_events.append({
                "timestamp": zc["timestamp"],
                "type": "rotation",
                "description": f"Rotacionou de {zc['from_zone']} para {zc['to_zone']}",
            })

        # Score calculation
        # Good zone distribution (not camping spawn): 30 points
        # Spawn zones are now named "T Spawn" / "CT Spawn" in callout tables
        spawn_time = sum(
            pct for zname, pct in time_in_zones.items()
            if "spawn" in zname.lower()
        )
        zone_score = max(0, 30 - (spawn_time / 100) * 30)

        # Low exposed positioning: 30 points
        exposure_score = max(0, 30 - (exposed_pct / 100) * 30)

        # Appropriate rotation count (not too few, not too many): 20 points
        # Ideal is 3-8 rotations per match
        if 3 <= rotation_count <= 8:
            rotation_score = 20.0
        elif rotation_count < 3:
            rotation_score = rotation_count * 6.0
        else:
            rotation_score = max(0, 20 - (rotation_count - 8) * 2)

        # Fast rotations: 20 points (faster = better, ideal < 5 seconds)
        if avg_rotation_time > 0:
            rotation_speed_score = max(0, 20 - max(0, avg_rotation_time - 3) * 4)
        else:
            rotation_speed_score = 10.0

        total_score = zone_score + exposure_score + rotation_score + rotation_speed_score

        return MapAnalysisResult(
            positioning_score=round(total_score, 1),
            time_in_zones=time_in_zones,
            rotation_count=rotation_count,
            avg_rotation_time=round(avg_rotation_time, 1),
            exposed_positioning_pct=round(exposed_pct, 1),
            zone_timeline=zone_timeline,
            positioning_events=positioning_events,
            score=round(total_score, 1),
        )
