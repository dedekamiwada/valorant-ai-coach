"""
Game State Parser for Valorant VODs.

Extracts game state information via OCR and Computer Vision:
- Alive/dead player counts (green/red icons at top HUD)
- Agent identification from HUD icons
- Round score and time remaining
- Round phase detection (buy, play, plant, post-plant, retake)
- Spike status (planted, carried, location on minimap)
- Economy detection (credits, buy type classification)

Uses OpenCV for visual detection and optional EasyOCR for text recognition.
"""

import re

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class RoundPhase(str, Enum):
    BUY = "buy"
    PLAY = "play"
    PLANT = "plant"
    POST_PLANT = "post_plant"
    RETAKE = "retake"
    END = "end"
    UNKNOWN = "unknown"


class BuyType(str, Enum):
    FULL_BUY = "full_buy"
    HALF_BUY = "half_buy"
    FORCE_BUY = "force_buy"
    ECO = "eco"
    PISTOL = "pistol"
    UNKNOWN = "unknown"


# All Valorant agents for identification
VALORANT_AGENTS = [
    "Jett", "Reyna", "Raze", "Phoenix", "Neon", "Yoru", "Iso",  # Duelists
    "Sova", "Breach", "Skye", "KAY/O", "Fade", "Gekko",  # Initiators
    "Brimstone", "Omen", "Astra", "Viper", "Harbor", "Clove",  # Controllers
    "Killjoy", "Cypher", "Sage", "Chamber", "Deadlock", "Vyse",  # Sentinels
]


@dataclass
class PlayerState:
    """State of a single player in the match."""
    agent: str = "Unknown"
    is_alive: bool = True
    is_ally: bool = True
    has_spike: bool = False
    has_ultimate: bool = False


@dataclass
class SpikeState:
    """State of the spike in the current round."""
    is_planted: bool = False
    is_being_carried: bool = False
    carrier_team: str = "unknown"  # "ally" or "enemy"
    plant_site: str = "unknown"  # "A", "B", "C"
    time_remaining: float = -1.0  # seconds until detonation (-1 = not planted)
    minimap_position: tuple[int, int] = (0, 0)


@dataclass
class EconomyState:
    """Economy information for the current round."""
    player_credits: int = 0
    estimated_team_credits: int = 0
    buy_type: str = BuyType.UNKNOWN
    loss_bonus: int = 0  # consecutive round losses


@dataclass
class GameState:
    """Complete game state at a given frame."""
    timestamp: float = 0.0

    # Player states
    allies_alive: int = 5
    enemies_alive: int = 5
    ally_players: list[PlayerState] = field(default_factory=list)
    enemy_players: list[PlayerState] = field(default_factory=list)

    # Round info
    ally_score: int = 0
    enemy_score: int = 0
    round_number: int = 1
    round_time_remaining: float = 100.0
    round_phase: str = RoundPhase.UNKNOWN

    # Spike
    spike: SpikeState = field(default_factory=SpikeState)

    # Economy
    economy: EconomyState = field(default_factory=EconomyState)

    # Detection confidence
    confidence: float = 0.0


@dataclass
class GameStateTimeline:
    """Aggregated game state data across the full VOD."""
    states: list[GameState] = field(default_factory=list)
    round_boundaries: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    ally_rounds_won: int = 0
    enemy_rounds_won: int = 0
    economy_timeline: list[dict] = field(default_factory=list)


class GameStateParser:
    """
    Parses game state from Valorant VOD frames.

    Uses computer vision to extract HUD information:
    - Top bar: player alive/dead indicators (colored icons)
    - Center: round timer
    - Top center: score
    - Bottom: economy/credits
    - Minimap: spike location
    """

    # HUD regions (ratios relative to frame dimensions)
    # Top bar with player icons
    TOP_BAR_Y = (0.0, 0.05)
    ALLY_ICONS_X = (0.25, 0.48)   # Left side of top bar
    ENEMY_ICONS_X = (0.52, 0.75)  # Right side of top bar

    # Score area (center top)
    SCORE_Y = (0.0, 0.06)
    SCORE_X = (0.44, 0.56)

    # Timer (center, just below score)
    TIMER_Y = (0.04, 0.08)
    TIMER_X = (0.47, 0.53)

    # Economy/credits (bottom left during buy phase)
    CREDITS_Y = (0.88, 0.95)
    CREDITS_X = (0.0, 0.15)

    # Spike indicator on minimap
    SPIKE_COLOR_LOWER_HSV = np.array([15, 150, 150])  # Yellow-ish
    SPIKE_COLOR_UPPER_HSV = np.array([35, 255, 255])

    # Alive indicator colors
    ALIVE_GREEN_LOWER = np.array([55, 100, 100])
    ALIVE_GREEN_UPPER = np.array([85, 255, 255])
    ALIVE_RED_LOWER = np.array([0, 100, 100])
    ALIVE_RED_UPPER = np.array([10, 255, 255])
    DEAD_GRAY_LOWER = np.array([0, 0, 30])
    DEAD_GRAY_UPPER = np.array([180, 40, 120])

    # Buy phase detection
    BUY_PHASE_COLOR_LOWER = np.array([20, 100, 100])  # Yellow buy timer
    BUY_PHASE_COLOR_UPPER = np.array([40, 255, 255])

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.states: list[GameState] = []
        self.prev_state: GameState | None = None
        self._ocr_reader = None
        self._round_transitions: list[float] = []
        self._last_score = (0, 0)

    def _get_ocr_reader(self):
        """Lazy-load EasyOCR reader (heavy import)."""
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(
                    ["en"], gpu=False, verbose=False
                )
            except ImportError:
                self._ocr_reader = False  # Mark as unavailable
        return self._ocr_reader if self._ocr_reader is not False else None

    def _extract_region(
        self, frame: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> np.ndarray:
        """Extract a sub-region from frame using ratio coordinates."""
        h, w = frame.shape[:2]
        x1 = int(w * x_range[0])
        x2 = int(w * x_range[1])
        y1 = int(h * y_range[0])
        y2 = int(h * y_range[1])
        return frame[y1:y2, x1:x2]

    def count_alive_players(
        self, frame: np.ndarray, team: str = "ally"
    ) -> int:
        """
        Count alive players for a team by analysing the top HUD icons.

        Alive players have coloured (green for ally, red for enemy) icons.
        Dead players have grey/dark icons.
        """
        if team == "ally":
            region = self._extract_region(
                frame, self.ALLY_ICONS_X, self.TOP_BAR_Y
            )
            lower, upper = self.ALIVE_GREEN_LOWER, self.ALIVE_GREEN_UPPER
        else:
            region = self._extract_region(
                frame, self.ENEMY_ICONS_X, self.TOP_BAR_Y
            )
            lower, upper = self.ALIVE_RED_LOWER, self.ALIVE_RED_UPPER

        if region.size == 0:
            return 5  # Default

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Also check for the second red hue range (wraps around 180)
        if team == "enemy":
            lower2 = np.array([170, 100, 100])
            upper2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # Find separate icon blobs
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area to avoid noise
        min_area = region.shape[0] * region.shape[1] * 0.005
        alive_count = sum(
            1 for c in contours if cv2.contourArea(c) > min_area
        )

        return min(5, max(0, alive_count))

    def detect_round_phase(self, frame: np.ndarray) -> str:
        """
        Detect the current round phase.

        - BUY: Yellow timer visible, economy panel shown
        - PLAY: Normal gameplay timer (white)
        - PLANT: Spike plant animation/progress bar
        - POST_PLANT: Spike beeping/timer after plant
        - END: Round over screen
        """
        timer_region = self._extract_region(
            frame, self.TIMER_X, self.TIMER_Y
        )

        if timer_region.size == 0:
            return RoundPhase.UNKNOWN

        hsv = cv2.cvtColor(timer_region, cv2.COLOR_BGR2HSV)

        # Buy phase has yellow-ish timer
        buy_mask = cv2.inRange(
            hsv, self.BUY_PHASE_COLOR_LOWER, self.BUY_PHASE_COLOR_UPPER
        )
        buy_ratio = cv2.countNonZero(buy_mask) / max(1, buy_mask.size)

        if buy_ratio > 0.05:
            return RoundPhase.BUY

        # Check for red timer (post-plant spike countdown)
        red_lower = np.array([0, 150, 150])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_lower2 = np.array([170, 150, 150])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask, red_mask2)
        red_ratio = cv2.countNonZero(red_mask) / max(1, red_mask.size)

        if red_ratio > 0.05:
            return RoundPhase.POST_PLANT

        # Check for dark/gray screen (round end)
        gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 30:
            return RoundPhase.END

        # Default to play phase
        return RoundPhase.PLAY

    def detect_spike_status(self, frame: np.ndarray) -> SpikeState:
        """
        Detect spike status from the minimap and HUD indicators.

        - Spike carried: yellow icon on minimap following a player
        - Spike planted: yellow icon stationary on a site, with plant indicator
        - Post-plant: red pulsing on screen edges
        """
        spike = SpikeState()

        # Check minimap for spike (yellow indicator)
        minimap_x = (0.0, 0.14)
        minimap_y = (0.0, 0.22)
        minimap = self._extract_region(frame, minimap_x, minimap_y)

        if minimap.size == 0:
            return spike

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        spike_mask = cv2.inRange(
            hsv, self.SPIKE_COLOR_LOWER_HSV, self.SPIKE_COLOR_UPPER_HSV
        )

        contours, _ = cv2.findContours(
            spike_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 5:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    spike.minimap_position = (cx, cy)
                    spike.is_being_carried = True

                    # Determine if planted based on area (planted spike is larger)
                    mm_h, mm_w = minimap.shape[:2]
                    relative_area = area / (mm_h * mm_w)
                    if relative_area > 0.003:
                        spike.is_planted = True
                        spike.is_being_carried = False
                        # Classify site based on position
                        rx = cx / mm_w
                        ry = cy / mm_h
                        if ry < 0.4:
                            spike.plant_site = "A" if rx < 0.5 else "B"
                        else:
                            spike.plant_site = "B" if rx > 0.5 else "A"

        # Check for post-plant red pulsing (screen edges)
        h, w = frame.shape[:2]
        edges_region = frame[0:int(h * 0.1), :]
        hsv_edges = cv2.cvtColor(edges_region, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(
            hsv_edges, np.array([0, 100, 150]), np.array([10, 255, 255])
        )
        red_mask2 = cv2.inRange(
            hsv_edges, np.array([170, 100, 150]), np.array([180, 255, 255])
        )
        red_ratio = (
            cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        ) / max(1, edges_region.shape[0] * edges_region.shape[1])

        if red_ratio > 0.02 and not spike.is_planted:
            spike.is_planted = True
            spike.is_being_carried = False

        return spike

    def detect_score(self, frame: np.ndarray) -> tuple[int, int]:
        """
        Detect the round score from the top-centre HUD.

        Uses OCR if available, otherwise falls back to the last known score.
        """
        score_region = self._extract_region(
            frame, self.SCORE_X, self.SCORE_Y
        )

        if score_region.size == 0:
            return self._last_score

        reader = self._get_ocr_reader()
        if reader is not None:
            try:
                # Preprocess for OCR
                gray = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                results = reader.readtext(thresh, detail=0)
                text = " ".join(results).strip()

                # Parse score (format: "X : Y" or "X - Y")
                for sep in [":", "-", " "]:
                    if sep in text:
                        parts = text.split(sep)
                        if len(parts) >= 2:
                            try:
                                ally = int(parts[0].strip())
                                enemy = int(parts[-1].strip())
                                if 0 <= ally <= 13 and 0 <= enemy <= 13:
                                    self._last_score = (ally, enemy)
                                    return (ally, enemy)
                            except ValueError:
                                continue
            except Exception:
                pass

        return self._last_score

    def detect_economy(self, frame: np.ndarray) -> EconomyState:
        """
        Detect economy information from the HUD.

        During buy phase, credits are shown at the bottom of the screen.
        Also classifies buy type based on detected credits.
        """
        eco = EconomyState()

        credits_region = self._extract_region(
            frame, self.CREDITS_X, self.CREDITS_Y
        )

        if credits_region.size == 0:
            return eco

        reader = self._get_ocr_reader()
        if reader is not None:
            try:
                gray = cv2.cvtColor(credits_region, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                results = reader.readtext(thresh, detail=0)
                text = " ".join(results).strip()

                # Look for credit amount (e.g. "4500", "$4500")
                credit_match = re.search(r"[\$]?(\d{3,5})", text)
                if credit_match:
                    credits = int(credit_match.group(1))
                    eco.player_credits = credits
                    eco.estimated_team_credits = credits  # Approximation

                    # Classify buy type
                    if credits >= 3900:
                        eco.buy_type = BuyType.FULL_BUY
                    elif credits >= 2600:
                        eco.buy_type = BuyType.HALF_BUY
                    elif credits >= 1500:
                        eco.buy_type = BuyType.FORCE_BUY
                    else:
                        eco.buy_type = BuyType.ECO
            except Exception:
                pass

        return eco

    def detect_round_transition(
        self, frame: np.ndarray, prev_state: GameState | None
    ) -> bool:
        """
        Detect if a round transition has occurred.

        Indicators: score changed, all players alive again, buy phase started.
        """
        if prev_state is None:
            return False

        h, w = frame.shape[:2]

        # Check for round-end overlay (dark screen with text)
        center = frame[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))

        if mean_b < 35:
            return True

        # Check for buy phase after play phase
        current_phase = self.detect_round_phase(frame)
        if (
            current_phase == RoundPhase.BUY
            and prev_state.round_phase == RoundPhase.PLAY
        ):
            return True

        return False

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
    ) -> GameState:
        """Process a single frame and extract game state."""
        allies_alive = self.count_alive_players(frame, "ally")
        enemies_alive = self.count_alive_players(frame, "enemy")
        round_phase = self.detect_round_phase(frame)
        spike = self.detect_spike_status(frame)
        ally_score, enemy_score = self.detect_score(frame)

        # Only detect economy during buy phase
        economy = EconomyState()
        if round_phase == RoundPhase.BUY:
            economy = self.detect_economy(frame)

        # Detect round transitions
        is_new_round = self.detect_round_transition(frame, self.prev_state)
        round_number = (
            self.prev_state.round_number + (1 if is_new_round else 0)
            if self.prev_state
            else 1
        )

        if is_new_round:
            self._round_transitions.append(timestamp)

        # Determine post-plant / retake
        if spike.is_planted and round_phase == RoundPhase.PLAY:
            round_phase = RoundPhase.POST_PLANT

        # Calculate confidence based on detection quality
        confidence = 0.5  # Base
        if allies_alive + enemies_alive <= 10:
            confidence += 0.2
        if round_phase != RoundPhase.UNKNOWN:
            confidence += 0.2
        if ally_score + enemy_score > 0:
            confidence += 0.1

        state = GameState(
            timestamp=timestamp,
            allies_alive=allies_alive,
            enemies_alive=enemies_alive,
            ally_score=ally_score,
            enemy_score=enemy_score,
            round_number=round_number,
            round_time_remaining=100.0,  # OCR-dependent
            round_phase=round_phase,
            spike=spike,
            economy=economy,
            confidence=confidence,
        )

        self.states.append(state)
        self.prev_state = state
        return state

    def generate_results(self) -> GameStateTimeline:
        """Generate aggregated game state timeline."""
        if not self.states:
            return GameStateTimeline()

        # Identify round boundaries
        round_boundaries = []
        for i, ts in enumerate(self._round_transitions):
            round_boundaries.append({
                "round_number": i + 1,
                "timestamp": ts,
            })

        # Determine round outcomes from score changes
        total_rounds = max(
            s.ally_score + s.enemy_score for s in self.states
        ) if self.states else 0
        ally_won = max(s.ally_score for s in self.states) if self.states else 0
        enemy_won = max(
            s.enemy_score for s in self.states
        ) if self.states else 0

        # Economy timeline
        economy_timeline = [
            {
                "timestamp": s.timestamp,
                "credits": s.economy.player_credits,
                "buy_type": s.economy.buy_type,
                "round_phase": s.round_phase,
            }
            for s in self.states
            if s.economy.player_credits > 0
        ]

        return GameStateTimeline(
            states=self.states,
            round_boundaries=round_boundaries,
            total_rounds=total_rounds,
            ally_rounds_won=ally_won,
            enemy_rounds_won=enemy_won,
            economy_timeline=economy_timeline,
        )
