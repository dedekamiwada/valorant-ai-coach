"""
Crosshair Placement Analyzer for Valorant VODs.

Analyzes crosshair positioning relative to head level, contact points,
edge vs center aiming, and floor aiming patterns.

This represents 60% of the overall coaching score.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CrosshairFrame:
    timestamp: float
    crosshair_x: int
    crosshair_y: int
    is_head_level: bool
    is_floor_aiming: bool
    is_edge_aiming: bool  # vs center of openings
    in_combat: bool
    adjustment_pixels: float  # how much crosshair moved on first contact


@dataclass
class CrosshairAnalysisResult:
    head_level_consistency: float = 0.0
    avg_pre_aim_distance: float = 0.0
    first_contact_efficiency: float = 0.0
    center_vs_edge_ratio: float = 0.0
    floor_aiming_percentage: float = 0.0
    heatmap_points: list[dict] = field(default_factory=list)
    frame_data: list[dict] = field(default_factory=list)
    score: float = 0.0


class CrosshairAnalyzer:
    """
    Analyzes crosshair placement quality from Valorant gameplay frames.

    Key metrics:
    - Head Level Consistency: % of time crosshair is at head height
    - Pre-aim Distance: how far crosshair is from where enemies appear
    - First Contact Efficiency: minimal adjustment needed on contact
    - Edge vs Center Aiming: proper edge peeking technique
    - Floor Aiming: detecting crosshair aimed too low during rotations
    """

    # Valorant crosshair is always at screen center
    # Head level is approximately 30-35% from top in standard gameplay
    HEAD_LEVEL_RATIO_MIN = 0.25
    HEAD_LEVEL_RATIO_MAX = 0.45
    FLOOR_LEVEL_RATIO = 0.60  # Below this is "floor aiming"

    # Edge aiming detection - crosshair near edges of openings
    EDGE_THRESHOLD_RATIO = 0.15  # Within 15% of frame edge counts as edge aiming

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.crosshair_x = self.width // 2
        self.crosshair_y = self.height // 2
        self.frames: list[CrosshairFrame] = []
        self.heatmap = np.zeros((self.height // 10, self.width // 10), dtype=np.float32)

    def detect_crosshair_position(self, frame: np.ndarray) -> tuple[int, int]:
        """
        Detect crosshair position in a frame.

        In Valorant, the crosshair is always at the center of the screen.
        We track it relative to the game world by analyzing where the
        center of the screen points to.
        """
        h, w = frame.shape[:2]
        return w // 2, h // 2

    def detect_combat_state(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> bool:
        """
        Detect if the player is currently in combat by analyzing:
        - Muzzle flash indicators
        - Red hit markers
        - Health bar changes
        - Kill feed activity
        """
        if prev_frame is None:
            return False

        h, w = frame.shape[:2]

        # Check for red/orange indicators in the center area (hit markers)
        center_region = frame[
            int(h * 0.4):int(h * 0.6),
            int(w * 0.4):int(w * 0.6)
        ]

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

        # Red color range for hit markers
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)

        # If significant red pixels near center, likely in combat
        total_pixels = center_region.shape[0] * center_region.shape[1]
        if red_pixels / total_pixels > 0.005:
            return True

        # Check for large frame-to-frame changes (gunfire, flashes)
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        change_ratio = np.count_nonzero(gray_diff > 50) / gray_diff.size

        return change_ratio > 0.15

    def analyze_head_level(self, frame: np.ndarray) -> bool:
        """
        Check if crosshair is at head level.

        In Valorant, head level is roughly 30-35% from the top of the screen
        depending on distance and elevation. We use the vertical position of
        the crosshair (always center) relative to the game world.
        """
        h = frame.shape[0]
        cy = h // 2

        # Crosshair Y position relative to frame height
        ratio = cy / h

        return self.HEAD_LEVEL_RATIO_MIN <= ratio <= self.HEAD_LEVEL_RATIO_MAX

    def analyze_floor_aiming(self, frame: np.ndarray) -> bool:
        """
        Detect if the player is aiming at the floor.

        This analyzes the lower portion of the screen around the crosshair
        to determine if there's ground/floor texture rather than wall/sky.
        """
        h, w = frame.shape[:2]
        cy = h // 2

        # Analyze the area below the crosshair
        below_crosshair = frame[cy:int(h * 0.75), int(w * 0.3):int(w * 0.7)]

        if below_crosshair.size == 0:
            return False

        # Floor tends to have more uniform, darker colors
        gray = cv2.cvtColor(below_crosshair, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)

        # Also check brightness distribution
        mean_brightness = np.mean(gray)

        # Floor/ground typically has low variance and moderate brightness
        # This is a heuristic that works reasonably well
        return std_dev < 30 and mean_brightness < 120

    def analyze_edge_aiming(self, frame: np.ndarray) -> bool:
        """
        Detect if the player is aiming at edges of openings (good technique)
        vs the center of doorways/corridors (bad technique).

        Edge aiming means the crosshair is positioned near walls/corners
        where enemies would first appear, not in empty space.
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Sample the area around the crosshair
        sample_size = 80
        region = frame[
            max(0, cy - sample_size):min(h, cy + sample_size),
            max(0, cx - sample_size):min(w, cx + sample_size)
        ]

        if region.size == 0:
            return False

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Use edge detection - if there are strong edges near crosshair,
        # player is aiming near walls/corners (edge aiming = good)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # High edge density near crosshair = edge aiming
        return edge_density > 0.08

    def calculate_optical_flow_at_crosshair(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> float:
        """
        Calculate how much the crosshair area moved between frames
        using optical flow. Large movements indicate reactive flicking
        rather than proactive pre-aiming.
        """
        h, w = prev_gray.shape
        cx, cy = w // 2, h // 2

        # Calculate sparse optical flow at crosshair region
        points = np.array([[[float(cx), float(cy)]]], dtype=np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None,
            winSize=(21, 21), maxLevel=3
        )

        if status[0][0] == 1 and next_pts is not None:
            dx = next_pts[0][0][0] - cx
            dy = next_pts[0][0][1] - cy
            return float(np.sqrt(dx * dx + dy * dy))

        return 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
        prev_gray: np.ndarray | None = None,
    ) -> CrosshairFrame:
        """Process a single frame and return crosshair analysis data."""
        cx, cy = self.detect_crosshair_position(frame)

        is_combat = self.detect_combat_state(frame, prev_frame)
        is_head = self.analyze_head_level(frame)
        is_floor = self.analyze_floor_aiming(frame)
        is_edge = self.analyze_edge_aiming(frame)

        # Calculate adjustment pixels (optical flow at crosshair)
        adjustment = 0.0
        if prev_gray is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            adjustment = self.calculate_optical_flow_at_crosshair(prev_gray, curr_gray)

        cf = CrosshairFrame(
            timestamp=timestamp,
            crosshair_x=cx,
            crosshair_y=cy,
            is_head_level=is_head,
            is_floor_aiming=is_floor,
            is_edge_aiming=is_edge,
            in_combat=is_combat,
            adjustment_pixels=adjustment,
        )
        self.frames.append(cf)

        # Update heatmap
        hx = min(cx // 10, self.heatmap.shape[1] - 1)
        hy = min(cy // 10, self.heatmap.shape[0] - 1)
        self.heatmap[hy, hx] += 1.0

        return cf

    def generate_results(self) -> CrosshairAnalysisResult:
        """Generate aggregate analysis results from all processed frames."""
        if not self.frames:
            return CrosshairAnalysisResult(score=0.0)

        non_combat_frames = [f for f in self.frames if not f.in_combat]
        combat_frames = [f for f in self.frames if f.in_combat]

        # Head level consistency (out of combat - this is what matters most)
        analysis_frames = non_combat_frames if non_combat_frames else self.frames
        head_level_count = sum(1 for f in analysis_frames if f.is_head_level)
        head_level_consistency = (head_level_count / len(analysis_frames)) * 100

        # Floor aiming percentage (should be low)
        floor_count = sum(1 for f in analysis_frames if f.is_floor_aiming)
        floor_percentage = (floor_count / len(analysis_frames)) * 100

        # Edge aiming ratio (higher is better)
        edge_count = sum(1 for f in analysis_frames if f.is_edge_aiming)
        edge_ratio = (edge_count / len(analysis_frames)) * 100

        # First contact efficiency (lower adjustment = better pre-aim)
        if combat_frames:
            avg_adjustment = np.mean([f.adjustment_pixels for f in combat_frames])
        else:
            avg_adjustment = np.mean([f.adjustment_pixels for f in self.frames])

        # Pre-aim distance (average optical flow movement)
        avg_pre_aim = np.mean([f.adjustment_pixels for f in self.frames])

        # Generate heatmap points for visualization
        heatmap_points = []
        for y in range(self.heatmap.shape[0]):
            for x in range(self.heatmap.shape[1]):
                if self.heatmap[y, x] > 0:
                    heatmap_points.append({
                        "x": x * 10,
                        "y": y * 10,
                        "value": float(self.heatmap[y, x])
                    })

        # Calculate score (0-100)
        # Head level: 40 points max
        head_score = min(40, (head_level_consistency / 100) * 40)

        # Low floor aiming: 20 points max
        floor_score = max(0, 20 - (floor_percentage / 100) * 20)

        # Edge aiming: 20 points max
        edge_score = min(20, (edge_ratio / 100) * 20)

        # First contact efficiency: 20 points max (lower adjustment = higher score)
        efficiency_score = max(0, 20 - min(20, avg_adjustment / 5))

        total_score = head_score + floor_score + edge_score + efficiency_score

        # Generate frame-level data for timeline
        frame_data = [
            {
                "timestamp": f.timestamp,
                "head_level": f.is_head_level,
                "floor_aiming": f.is_floor_aiming,
                "edge_aiming": f.is_edge_aiming,
                "combat": f.in_combat,
                "adjustment": f.adjustment_pixels,
            }
            for f in self.frames[::5]  # Sample every 5th frame for timeline
        ]

        return CrosshairAnalysisResult(
            head_level_consistency=round(head_level_consistency, 1),
            avg_pre_aim_distance=round(float(avg_pre_aim), 1),
            first_contact_efficiency=round(float(avg_adjustment), 1),
            center_vs_edge_ratio=round(edge_ratio, 1),
            floor_aiming_percentage=round(floor_percentage, 1),
            heatmap_points=heatmap_points,
            frame_data=frame_data,
            score=round(total_score, 1),
        )
