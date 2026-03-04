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


class MapAnalyzer:
    """
    Analyzes player positioning from the Valorant minimap.

    The minimap in Valorant is located in the top-left corner of the screen.
    It shows player positions, teammate positions, and the map layout.

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

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
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

    def classify_zone(self, position: tuple[int, int], minimap_size: tuple[int, int]) -> str:
        """
        Classify the player's position into a map zone.

        Divides the minimap into logical zones based on typical Valorant map layouts.
        """
        w, h = minimap_size
        if w == 0 or h == 0:
            return "unknown"

        rx = position[0] / w  # relative x (0-1)
        ry = position[1] / h  # relative y (0-1)

        # General zone classification based on minimap regions
        # Most Valorant maps have A site on one side and B on the other
        if ry < 0.25:
            if rx < 0.4:
                return "a_site"
            elif rx > 0.6:
                return "b_site"
            else:
                return "mid"
        elif ry < 0.5:
            if rx < 0.35:
                return "a_main"
            elif rx > 0.65:
                return "b_main"
            else:
                return "mid"
        elif ry < 0.75:
            if rx < 0.3:
                return "a_lobby"
            elif rx > 0.7:
                return "b_lobby"
            else:
                return "mid"
        else:
            return "spawn"

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
        aggressive_zones = {"a_site", "b_site", "a_main", "b_main"}
        if zone in aggressive_zones and teammates_nearby < 1:
            return True

        # Mid with no support is risky
        if zone == "mid" and teammates_nearby < 1:
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
        spawn_time = time_in_zones.get("spawn", 0)
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
