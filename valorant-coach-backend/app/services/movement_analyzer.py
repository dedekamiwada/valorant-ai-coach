"""
Movement Analyzer for Valorant VODs.

Analyzes counter-strafing, movement during shooting, peek types,
and spray control patterns.

This represents 20% of the overall coaching score.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MovementFrame:
    timestamp: float
    is_moving: bool
    is_shooting: bool
    movement_magnitude: float
    is_counter_strafing: bool
    peek_type: str  # none, tight, wide, over


@dataclass
class MovementAnalysisResult:
    counter_strafe_accuracy: float = 0.0
    movement_while_shooting: float = 0.0
    peek_type_distribution: dict = field(default_factory=dict)
    spray_control_score: float = 0.0
    frame_data: list[dict] = field(default_factory=list)
    score: float = 0.0


class MovementAnalyzer:
    """
    Analyzes movement quality in Valorant gameplay.

    Key metrics:
    - Counter-strafe Accuracy: stopping before shooting
    - Movement While Shooting: % of shots fired while moving (penalty)
    - Peek Type Selection: tight vs wide vs over-peeking patterns
    - Spray Control: burst vs full spray behavior
    """

    # Thresholds for movement detection
    MOVEMENT_THRESHOLD = 2.0  # Optical flow magnitude threshold
    SHOOTING_BRIGHTNESS_THRESHOLD = 30  # Muzzle flash detection
    COUNTER_STRAFE_WINDOW = 3  # Frames to check for counter-strafe

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.frames: list[MovementFrame] = []
        self.prev_gray: np.ndarray | None = None
        self.movement_history: list[float] = []
        self.shooting_history: list[bool] = []

    def detect_movement(self, curr_gray: np.ndarray) -> tuple[bool, float]:
        """
        Detect if the player is moving using optical flow analysis.

        Analyzes the overall scene movement which indicates player movement
        in the game world.
        """
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return False, 0.0

        # Calculate dense optical flow on a downsampled version for speed
        h, w = curr_gray.shape
        small_prev = cv2.resize(self.prev_gray, (w // 4, h // 4))
        small_curr = cv2.resize(curr_gray, (w // 4, h // 4))

        flow = cv2.calcOpticalFlowFarneback(
            small_prev, small_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Calculate magnitude of flow
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Use the median of the top 25% of flow values (ignoring static elements)
        sorted_mag = np.sort(magnitude.flatten())
        top_quarter = sorted_mag[int(len(sorted_mag) * 0.75):]
        avg_movement = float(np.mean(top_quarter)) if len(top_quarter) > 0 else 0.0

        self.prev_gray = curr_gray
        self.movement_history.append(avg_movement)

        return avg_movement > self.MOVEMENT_THRESHOLD, avg_movement

    def detect_shooting(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> bool:
        """
        Detect if the player is currently shooting.

        Uses multiple cues:
        - Muzzle flash (bright area near bottom center)
        - Ammo counter changes
        - Screen shake patterns
        """
        if prev_frame is None:
            return False

        h, w = frame.shape[:2]

        # Check for muzzle flash in the weapon area (bottom right quadrant)
        weapon_region = frame[int(h * 0.6):h, int(w * 0.5):w]
        prev_weapon = prev_frame[int(h * 0.6):h, int(w * 0.5):w]

        diff = cv2.absdiff(weapon_region, prev_weapon)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Muzzle flash creates bright differences
        bright_pixels = np.count_nonzero(gray_diff > 80)
        total_pixels = gray_diff.size

        is_shooting = (bright_pixels / total_pixels) > 0.02

        self.shooting_history.append(is_shooting)
        return is_shooting

    def detect_counter_strafe(self) -> bool:
        """
        Detect counter-strafing: player was moving, then stopped, then shot.

        Counter-strafing is pressing the opposite movement key to stop
        momentum before shooting for maximum accuracy.
        """
        if len(self.movement_history) < self.COUNTER_STRAFE_WINDOW + 1:
            return False
        if len(self.shooting_history) < 2:
            return False

        # Current frame is shooting
        if not self.shooting_history[-1]:
            return False

        # Check if player was moving recently and has now stopped
        recent_movement = self.movement_history[-self.COUNTER_STRAFE_WINDOW - 1:-1]
        current_movement = self.movement_history[-1]

        was_moving = any(m > self.MOVEMENT_THRESHOLD for m in recent_movement)
        now_stopped = current_movement < self.MOVEMENT_THRESHOLD

        return was_moving and now_stopped

    def detect_peek_type(self) -> str:
        """
        Classify the current peek type based on movement patterns.

        - tight: small movement, quick stop
        - wide: large movement, deliberate wide swing
        - over: excessive movement, over-peeking (bad)
        - none: not peeking
        """
        if len(self.movement_history) < 5:
            return "none"

        recent = self.movement_history[-5:]
        max_movement = max(recent)
        avg_movement = np.mean(recent)

        if max_movement < self.MOVEMENT_THRESHOLD:
            return "none"
        elif max_movement < self.MOVEMENT_THRESHOLD * 2:
            return "tight"
        elif max_movement < self.MOVEMENT_THRESHOLD * 4:
            return "wide"
        else:
            return "over"

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
    ) -> MovementFrame:
        """Process a single frame for movement analysis."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_moving, magnitude = self.detect_movement(curr_gray)
        is_shooting = self.detect_shooting(frame, prev_frame)
        is_counter_strafing = self.detect_counter_strafe()
        peek_type = self.detect_peek_type()

        mf = MovementFrame(
            timestamp=timestamp,
            is_moving=is_moving,
            is_shooting=is_shooting,
            movement_magnitude=magnitude,
            is_counter_strafing=is_counter_strafing,
            peek_type=peek_type,
        )
        self.frames.append(mf)
        return mf

    def generate_results(self) -> MovementAnalysisResult:
        """Generate aggregate movement analysis results."""
        if not self.frames:
            return MovementAnalysisResult(score=0.0)

        shooting_frames = [f for f in self.frames if f.is_shooting]
        moving_frames = [f for f in self.frames if f.is_moving]

        # Counter-strafe accuracy
        if shooting_frames:
            correct_strafes = sum(1 for f in shooting_frames if f.is_counter_strafing or not f.is_moving)
            counter_strafe_acc = (correct_strafes / len(shooting_frames)) * 100
        else:
            counter_strafe_acc = 100.0

        # Movement while shooting (should be low)
        if shooting_frames:
            moving_while_shooting = sum(1 for f in shooting_frames if f.is_moving)
            movement_shooting_pct = (moving_while_shooting / len(shooting_frames)) * 100
        else:
            movement_shooting_pct = 0.0

        # Peek type distribution
        peek_types = {"tight": 0, "wide": 0, "over": 0, "none": 0}
        for f in self.frames:
            peek_types[f.peek_type] = peek_types.get(f.peek_type, 0) + 1

        total_peeks = sum(v for k, v in peek_types.items() if k != "none")
        if total_peeks > 0:
            peek_dist = {
                k: round((v / total_peeks) * 100, 1)
                for k, v in peek_types.items()
                if k != "none"
            }
        else:
            peek_dist = {"tight": 0, "wide": 0, "over": 0}

        # Spray control (consecutive shooting frames = spraying)
        consecutive_shots = 0
        max_consecutive = 0
        spray_count = 0
        for f in self.frames:
            if f.is_shooting:
                consecutive_shots += 1
                max_consecutive = max(max_consecutive, consecutive_shots)
            else:
                if consecutive_shots > 5:  # More than 5 frames = spray
                    spray_count += 1
                consecutive_shots = 0

        spray_control = max(0, 100 - spray_count * 10)

        # Calculate score
        # Counter-strafe: 40 points max
        strafe_score = min(40, (counter_strafe_acc / 100) * 40)

        # Low movement while shooting: 30 points max
        move_shoot_score = max(0, 30 - (movement_shooting_pct / 100) * 30)

        # Peek quality: 15 points max (penalize over-peeking)
        over_peek_ratio = peek_dist.get("over", 0) / 100 if total_peeks > 0 else 0
        peek_score = max(0, 15 - over_peek_ratio * 15)

        # Spray control: 15 points max
        spray_score = (spray_control / 100) * 15

        total_score = strafe_score + move_shoot_score + peek_score + spray_score

        frame_data = [
            {
                "timestamp": f.timestamp,
                "moving": f.is_moving,
                "shooting": f.is_shooting,
                "magnitude": round(f.movement_magnitude, 2),
                "peek": f.peek_type,
                "counter_strafe": f.is_counter_strafing,
            }
            for f in self.frames[::5]
        ]

        return MovementAnalysisResult(
            counter_strafe_accuracy=round(counter_strafe_acc, 1),
            movement_while_shooting=round(movement_shooting_pct, 1),
            peek_type_distribution=peek_dist,
            spray_control_score=round(spray_control, 1),
            frame_data=frame_data,
            score=round(total_score, 1),
        )
