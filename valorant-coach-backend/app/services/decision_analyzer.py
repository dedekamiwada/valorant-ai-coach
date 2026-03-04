"""
Decision Analyzer for Valorant VODs.

Analyzes tactical decision-making: angle exposure,
trade efficiency, utility impact, and commitment clarity.

This represents 15% of the overall coaching score.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DecisionFrame:
    timestamp: float
    exposed_angles: int  # how many angles can see the player
    is_using_cover: bool
    is_utility_used: bool
    utility_type: str  # none, smoke, flash, molly, wall, other


@dataclass
class DecisionAnalysisResult:
    multi_angle_exposure_count: int = 0
    trade_efficiency: float = 0.0
    utility_impact_score: float = 0.0
    commitment_clarity: float = 0.0
    exposure_timeline: list[dict] = field(default_factory=list)
    utility_events: list[dict] = field(default_factory=list)
    score: float = 0.0


class DecisionAnalyzer:
    """
    Analyzes tactical decision quality in Valorant gameplay.

    Key metrics:
    - Multi-angle Exposure: times exposed to multiple angles (bad)
    - Trade Efficiency: deaths that were properly traded by teammates
    - Utility Impact: whether abilities had meaningful impact
    - Commitment Clarity: clear fight/flee decisions vs hesitation
    """

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.frames: list[DecisionFrame] = []
        self.death_timestamps: list[float] = []
        self.kill_timestamps: list[float] = []
        self.utility_timestamps: list[dict] = []

    def detect_utility_usage(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> tuple[bool, str]:
        """
        Detect if a utility/ability is being used.

        Looks for:
        - Smoke particles (gray/blue translucent areas)
        - Flash effects (bright white screen)
        - Molly/fire effects (orange/red areas on ground)
        - Ability UI indicators
        """
        if prev_frame is None:
            return False, "none"

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Flash detection: sudden brightness increase
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        brightness_diff = float(np.mean(gray)) - float(np.mean(prev_gray))
        if brightness_diff > 60:
            return True, "flash"

        # Smoke detection: large areas of gray/blue translucency
        lower_smoke = np.array([90, 10, 100])
        upper_smoke = np.array([130, 80, 220])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        smoke_ratio = cv2.countNonZero(smoke_mask) / smoke_mask.size

        if smoke_ratio > 0.1:
            return True, "smoke"

        # Molly/fire detection: orange/red areas
        lower_fire = np.array([5, 150, 150])
        upper_fire = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = cv2.countNonZero(fire_mask) / fire_mask.size

        if fire_ratio > 0.05:
            return True, "molly"

        return False, "none"

    def detect_kill_or_death(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> tuple[bool, bool]:
        """
        Detect kill feed events and death screen.

        Kill: red/green indicators in top-right kill feed
        Death: dark overlay with respawn timer
        """
        if prev_frame is None:
            return False, False

        h, w = frame.shape[:2]

        # Kill feed area (top right)
        kill_feed = frame[0:int(h * 0.3), int(w * 0.6):w]
        prev_kill_feed = prev_frame[0:int(h * 0.3), int(w * 0.6):w]

        diff = cv2.absdiff(kill_feed, prev_kill_feed)
        change = np.mean(diff)

        is_kill = change > 20

        # Death detection: screen goes dark/gray
        center = frame[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        is_death = float(np.mean(gray)) < 40

        return is_kill, is_death

    def estimate_exposed_angles(self, frame: np.ndarray) -> int:
        """
        Estimate how many angles the player is exposed to.

        Uses edge detection and open space analysis around the crosshair
        to estimate exposure. More open space = more exposed angles.
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check 8 directions from crosshair for "openness"
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        open_angles = 0
        check_distance = min(w, h) // 6

        for dx, dy in directions:
            # Check if there's a wall/edge in this direction
            end_x = cx + dx * check_distance
            end_y = cy + dy * check_distance

            if end_x < 0 or end_x >= w or end_y < 0 or end_y >= h:
                continue

            # Sample along the line
            steps = 20
            has_wall = False
            for step in range(1, steps):
                sx = int(cx + dx * check_distance * step / steps)
                sy = int(cy + dy * check_distance * step / steps)

                if 0 <= sx < w and 0 <= sy < h:
                    if edges[sy, sx] > 0:
                        has_wall = True
                        break

            if not has_wall:
                open_angles += 1

        # Convert to approximate angle count (1-4)
        if open_angles <= 2:
            return 1
        elif open_angles <= 4:
            return 2
        elif open_angles <= 6:
            return 3
        else:
            return 4

    def detect_cover_usage(self, frame: np.ndarray) -> bool:
        """
        Detect if the player is using cover.

        Cover = edges/walls close to the crosshair on at least one side.
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check for edges near the crosshair (within 100px)
        near_region = edges[
            max(0, cy - 100):min(h, cy + 100),
            max(0, cx - 100):min(w, cx + 100)
        ]

        edge_density = np.count_nonzero(near_region) / near_region.size if near_region.size > 0 else 0
        return edge_density > 0.05

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
    ) -> DecisionFrame:
        """Process a single frame for decision analysis."""
        exposed_angles = self.estimate_exposed_angles(frame)
        is_using_cover = self.detect_cover_usage(frame)
        is_utility, utility_type = self.detect_utility_usage(frame, prev_frame)
        is_kill, is_death = self.detect_kill_or_death(frame, prev_frame)

        if is_kill:
            self.kill_timestamps.append(timestamp)
        if is_death:
            self.death_timestamps.append(timestamp)
        if is_utility:
            self.utility_timestamps.append({"timestamp": timestamp, "type": utility_type})

        df = DecisionFrame(
            timestamp=timestamp,
            exposed_angles=exposed_angles,
            is_using_cover=is_using_cover,
            is_utility_used=is_utility,
            utility_type=utility_type,
        )
        self.frames.append(df)
        return df

    def generate_results(self) -> DecisionAnalysisResult:
        """Generate decision analysis results."""
        if not self.frames:
            return DecisionAnalysisResult(score=0.0)

        # Multi-angle exposure count
        multi_exposure = sum(1 for f in self.frames if f.exposed_angles > 2)
        multi_exposure_pct = (multi_exposure / len(self.frames)) * 100

        # Trade efficiency (deaths followed by teammate kill within 3s)
        trade_window = 3.0
        traded_deaths = 0
        total_deaths = len(self.death_timestamps)

        for death_time in self.death_timestamps:
            for kill_time in self.kill_timestamps:
                if 0 < (kill_time - death_time) < trade_window:
                    traded_deaths += 1
                    break

        trade_efficiency = (traded_deaths / total_deaths * 100) if total_deaths > 0 else 100.0

        # Utility impact score (having utility events is good)
        total_util = len(self.utility_timestamps)
        util_score = min(100, total_util * 15)

        # Commitment clarity (frames with cover = clear positioning)
        cover_frames = sum(1 for f in self.frames if f.is_using_cover)
        cover_pct = (cover_frames / len(self.frames)) * 100

        # Calculate overall score
        # Low multi-angle exposure: 35 points
        exposure_score = max(0, 35 - (multi_exposure_pct / 100) * 35)

        # Trade efficiency: 25 points
        trade_score = (trade_efficiency / 100) * 25

        # Utility usage: 20 points
        utility_score = (util_score / 100) * 20

        # Cover usage: 20 points
        cover_score = min(20, (cover_pct / 100) * 20)

        total_score = exposure_score + trade_score + utility_score + cover_score

        exposure_timeline = [
            {
                "timestamp": f.timestamp,
                "angles": f.exposed_angles,
                "cover": f.is_using_cover,
            }
            for f in self.frames[::5]
        ]

        utility_events = self.utility_timestamps

        return DecisionAnalysisResult(
            multi_angle_exposure_count=multi_exposure,
            trade_efficiency=round(trade_efficiency, 1),
            utility_impact_score=round(util_score, 1),
            commitment_clarity=round(cover_pct, 1),
            exposure_timeline=exposure_timeline,
            utility_events=utility_events,
            score=round(total_score, 1),
        )
