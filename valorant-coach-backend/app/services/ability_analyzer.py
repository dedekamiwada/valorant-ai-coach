"""
Ability Analyzer for Valorant VODs.

Dedicated analysis of ability/utility usage:
- Flash detection: pop time, direction, effectiveness (enemies blinded)
- Smoke detection: landing location, coverage area, duration
- Molly/Incendiary detection: area of effect, line-up quality
- Recon/Drone detection: information gathered
- Trap/Wire detection: placement quality
- Ultimate detection: activation, area of effect, impact
- Per-ability efficiency scoring

This module works alongside the DecisionAnalyzer but provides
much more granular ability-specific insights.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class AbilityType(str, Enum):
    FLASH = "flash"
    SMOKE = "smoke"
    MOLLY = "molly"
    WALL = "wall"
    RECON = "recon"
    TRAP = "trap"
    HEAL = "heal"
    ULTIMATE = "ultimate"
    UNKNOWN = "unknown"


@dataclass
class AbilityEvent:
    """A single ability usage event."""
    timestamp: float
    ability_type: str
    duration: float = 0.0  # how long the ability lasted
    effectiveness: float = 0.0  # 0-100 score
    description: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class FlashEvent(AbilityEvent):
    """Flash-specific event data."""
    pop_time: float = 0.0  # time from throw to pop
    enemies_flashed: int = 0  # estimated enemies affected
    self_flashed: bool = False  # did it flash teammates/self?
    direction: str = "unknown"  # left, right, overhead, bounce


@dataclass
class SmokeEvent(AbilityEvent):
    """Smoke-specific event data."""
    landing_zone: str = "unknown"  # callout where it landed
    blocks_los: bool = False  # does it block an important line of sight?
    duration_remaining: float = 0.0


@dataclass
class AbilityFrame:
    """Per-frame ability analysis data."""
    timestamp: float
    flash_detected: bool = False
    smoke_active: bool = False
    molly_active: bool = False
    wall_active: bool = False
    recon_active: bool = False
    ability_type: str = AbilityType.UNKNOWN
    screen_flash_intensity: float = 0.0  # brightness spike for flashes
    smoke_coverage_pct: float = 0.0  # % of screen covered by smoke
    fire_coverage_pct: float = 0.0  # % of screen covered by fire


@dataclass
class AbilityAnalysisResult:
    """Aggregated ability analysis results."""
    total_abilities_used: int = 0
    flash_count: int = 0
    smoke_count: int = 0
    molly_count: int = 0
    wall_count: int = 0
    recon_count: int = 0
    ultimate_count: int = 0
    flash_effectiveness: float = 0.0  # avg flash score
    smoke_effectiveness: float = 0.0  # avg smoke score
    molly_effectiveness: float = 0.0  # avg molly score
    overall_utility_score: float = 0.0
    ability_events: list[dict] = field(default_factory=list)
    ability_timeline: list[dict] = field(default_factory=list)
    efficiency_breakdown: dict = field(default_factory=dict)
    score: float = 0.0


class AbilityAnalyzer:
    """
    Analyses ability/utility usage in Valorant gameplay.

    Detection methods:
    - Flashes: sudden brightness spikes across the entire screen
    - Smokes: large grey/blue translucent areas appearing
    - Mollies: orange/red ground effects
    - Walls: large structural elements appearing (e.g. Sage wall, Viper wall)
    - Recon: dart/drone visual indicators
    - Ultimates: screen-wide effects, unique visual cues
    """

    # Flash detection thresholds
    FLASH_BRIGHTNESS_THRESHOLD = 55  # Mean brightness jump between frames
    FLASH_SCREEN_RATIO = 0.35  # % of screen that goes white for a flash
    FLASH_MIN_DURATION = 0.1  # seconds
    FLASH_MAX_DURATION = 3.5  # seconds

    # Smoke detection
    SMOKE_COLOR_LOWER = np.array([85, 5, 80])
    SMOKE_COLOR_UPPER = np.array([135, 90, 220])
    SMOKE_MIN_COVERAGE = 0.06  # 6% of screen

    # Molly/fire detection
    FIRE_COLOR_LOWER = np.array([3, 140, 140])
    FIRE_COLOR_UPPER = np.array([28, 255, 255])
    FIRE_MIN_COVERAGE = 0.02  # 2% of screen

    # Wall detection (sudden large edge appearance)
    WALL_EDGE_THRESHOLD = 0.15

    # Recon/dart detection (small bright blue/yellow indicator)
    RECON_COLOR_LOWER = np.array([95, 150, 150])
    RECON_COLOR_UPPER = np.array([115, 255, 255])

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.frames: list[AbilityFrame] = []
        self.events: list[AbilityEvent] = []
        self.prev_brightness: float = 0.0
        self._flash_start: float | None = None
        self._smoke_start: float | None = None
        self._molly_start: float | None = None
        self._flash_cooldown: float = 0.0  # prevent double-counting
        self._flash_peak_intensity: float = 0.0  # track peak brightness during flash

    def detect_flash(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, timestamp: float
    ) -> tuple[bool, float]:
        """
        Detect flash effects by analysing brightness spikes.

        Returns (is_flash, intensity).
        """
        if prev_frame is None:
            return False, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        curr_mean = float(np.mean(gray))
        prev_mean = float(np.mean(prev_gray))
        brightness_jump = curr_mean - prev_mean

        # Check if large portion of screen is very bright
        white_pixels = np.count_nonzero(gray > 220)
        white_ratio = white_pixels / gray.size

        is_flash = (
            brightness_jump > self.FLASH_BRIGHTNESS_THRESHOLD
            or white_ratio > self.FLASH_SCREEN_RATIO
        )

        # Cooldown to prevent re-triggering on the same flash
        if is_flash and timestamp - self._flash_cooldown < 1.0:
            is_flash = False

        if is_flash:
            self._flash_cooldown = timestamp

        return is_flash, max(brightness_jump, 0.0)

    def detect_smoke(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Detect active smoke by looking for translucent grey/blue areas.

        Returns (smoke_active, coverage_percentage).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.SMOKE_COLOR_LOWER, self.SMOKE_COLOR_UPPER)

        # Also check for darker smoke variants (Omen, Astra)
        dark_lower = np.array([0, 0, 60])
        dark_upper = np.array([180, 30, 150])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

        # Focus on the central play area (ignore HUD)
        h, w = frame.shape[:2]
        play_area_mask = np.zeros_like(mask)
        play_area_mask[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 255
        mask = cv2.bitwise_and(mask, play_area_mask)
        dark_mask = cv2.bitwise_and(dark_mask, play_area_mask)

        combined = cv2.bitwise_or(mask, dark_mask)
        coverage = cv2.countNonZero(combined) / max(1, combined.size)

        return coverage > self.SMOKE_MIN_COVERAGE, coverage

    def detect_molly(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Detect molly/incendiary effects on the ground.

        Returns (molly_active, coverage_percentage).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.FIRE_COLOR_LOWER, self.FIRE_COLOR_UPPER)

        # Fire is typically in the lower portion of the screen (ground level)
        h, w = frame.shape[:2]
        ground_mask = np.zeros_like(mask)
        ground_mask[int(h * 0.4):h, :] = 255
        mask = cv2.bitwise_and(mask, ground_mask)

        coverage = cv2.countNonZero(mask) / max(1, mask.size)

        return coverage > self.FIRE_MIN_COVERAGE, coverage

    def detect_wall(
        self, frame: np.ndarray, prev_frame: np.ndarray | None
    ) -> bool:
        """
        Detect wall abilities (Sage wall, Viper wall, Harbor wall).

        Looks for sudden appearance of large structural edges.
        """
        if prev_frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        edges_curr = cv2.Canny(gray, 50, 150)
        edges_prev = cv2.Canny(prev_gray, 50, 150)

        new_edges = cv2.subtract(edges_curr, edges_prev)
        new_edge_ratio = np.count_nonzero(new_edges) / max(1, new_edges.size)

        return new_edge_ratio > self.WALL_EDGE_THRESHOLD

    def detect_recon(self, frame: np.ndarray) -> bool:
        """
        Detect recon abilities (Sova dart, Fade haunt, etc.).

        Looks for the characteristic blue/cyan scan indicators.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.RECON_COLOR_LOWER, self.RECON_COLOR_UPPER)

        # Recon pings are typically small but bright
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bright_blobs = sum(
            1 for c in contours
            if 50 < cv2.contourArea(c) < 5000
        )

        return bright_blobs >= 2  # Multiple scan indicators = recon active

    def detect_ultimate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None
    ) -> bool:
        """
        Detect ultimate ability activation.

        Ultimates often cause dramatic visual changes:
        - Screen tint changes (Viper, Omen)
        - Large particle effects
        - Unique overlays
        """
        if prev_frame is None:
            return False

        # Check for dramatic colour shift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)

        hue_diff = float(np.mean(np.abs(
            hsv[:, :, 0].astype(float) - prev_hsv[:, :, 0].astype(float)
        )))
        sat_diff = float(np.mean(np.abs(
            hsv[:, :, 1].astype(float) - prev_hsv[:, :, 1].astype(float)
        )))

        # Large hue shift + saturation change = likely ult
        return hue_diff > 20 and sat_diff > 30

    def _score_flash(self, intensity: float, duration: float) -> float:
        """Score a flash event (0-100)."""
        # Good flashes: high intensity, moderate duration
        intensity_score = min(50, intensity / 2.0)
        duration_score = 50 if 0.5 <= duration <= 2.5 else 25
        return min(100, intensity_score + duration_score)

    def _score_smoke(self, coverage: float, duration: float) -> float:
        """Score a smoke event (0-100)."""
        # Good smokes cover meaningful area and last full duration
        coverage_score = min(60, coverage * 500)
        duration_score = min(40, duration * 10)
        return min(100, coverage_score + duration_score)

    def _score_molly(self, coverage: float) -> float:
        """Score a molly event (0-100)."""
        return min(100, coverage * 800)

    def process_frame(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None,
        timestamp: float,
    ) -> AbilityFrame:
        """Process a single frame for ability detection."""
        is_flash, flash_intensity = self.detect_flash(frame, prev_frame, timestamp)
        smoke_active, smoke_coverage = self.detect_smoke(frame)
        molly_active, molly_coverage = self.detect_molly(frame)
        wall_detected = self.detect_wall(frame, prev_frame)
        recon_active = self.detect_recon(frame)
        ult_detected = self.detect_ultimate(frame, prev_frame)

        # Determine primary ability type for this frame
        ability_type = AbilityType.UNKNOWN
        if is_flash:
            ability_type = AbilityType.FLASH
        elif ult_detected:
            ability_type = AbilityType.ULTIMATE
        elif wall_detected:
            ability_type = AbilityType.WALL
        elif molly_active:
            ability_type = AbilityType.MOLLY
        elif smoke_active:
            ability_type = AbilityType.SMOKE
        elif recon_active:
            ability_type = AbilityType.RECON

        # Track flash events (start/end)
        if is_flash and self._flash_start is None:
            self._flash_start = timestamp
            self._flash_peak_intensity = flash_intensity
        elif is_flash and self._flash_start is not None:
            # Update peak intensity while flash is active
            self._flash_peak_intensity = max(self._flash_peak_intensity, flash_intensity)
        elif not is_flash and self._flash_start is not None:
            duration = timestamp - self._flash_start
            if duration >= self.FLASH_MIN_DURATION:
                self.events.append(AbilityEvent(
                    timestamp=self._flash_start,
                    ability_type=AbilityType.FLASH,
                    duration=duration,
                    effectiveness=self._score_flash(self._flash_peak_intensity, duration),
                    description=f"Flash em {self._flash_start:.1f}s (duração: {duration:.1f}s)",
                ))
            self._flash_start = None
            self._flash_peak_intensity = 0.0

        # Track smoke events
        if smoke_active and self._smoke_start is None:
            self._smoke_start = timestamp
        elif not smoke_active and self._smoke_start is not None:
            duration = timestamp - self._smoke_start
            if duration > 0.5:
                self.events.append(AbilityEvent(
                    timestamp=self._smoke_start,
                    ability_type=AbilityType.SMOKE,
                    duration=duration,
                    effectiveness=self._score_smoke(smoke_coverage, duration),
                    description=f"Smoke em {self._smoke_start:.1f}s (cobertura: {smoke_coverage * 100:.0f}%)",
                ))
            self._smoke_start = None

        # Track molly events
        if molly_active and self._molly_start is None:
            self._molly_start = timestamp
        elif not molly_active and self._molly_start is not None:
            duration = timestamp - self._molly_start
            if duration > 0.3:
                self.events.append(AbilityEvent(
                    timestamp=self._molly_start,
                    ability_type=AbilityType.MOLLY,
                    duration=duration,
                    effectiveness=self._score_molly(molly_coverage),
                    description=f"Molly em {self._molly_start:.1f}s (área: {molly_coverage * 100:.0f}%)",
                ))
            self._molly_start = None

        # Track wall events (single frame detection)
        if wall_detected:
            self.events.append(AbilityEvent(
                timestamp=timestamp,
                ability_type=AbilityType.WALL,
                duration=0.0,
                effectiveness=60.0,  # Base score
                description=f"Wall detectada em {timestamp:.1f}s",
            ))

        # Track recon events
        if recon_active:
            # Avoid duplicates within 2 seconds
            recent_recon = [
                e for e in self.events
                if e.ability_type == AbilityType.RECON
                and abs(e.timestamp - timestamp) < 2.0
            ]
            if not recent_recon:
                self.events.append(AbilityEvent(
                    timestamp=timestamp,
                    ability_type=AbilityType.RECON,
                    duration=0.0,
                    effectiveness=70.0,  # Info gathering is valuable
                    description=f"Recon em {timestamp:.1f}s",
                ))

        # Track ultimate events
        if ult_detected:
            recent_ult = [
                e for e in self.events
                if e.ability_type == AbilityType.ULTIMATE
                and abs(e.timestamp - timestamp) < 5.0
            ]
            if not recent_ult:
                self.events.append(AbilityEvent(
                    timestamp=timestamp,
                    ability_type=AbilityType.ULTIMATE,
                    duration=0.0,
                    effectiveness=80.0,  # High-value ability
                    description=f"Ultimate ativada em {timestamp:.1f}s",
                ))

        af = AbilityFrame(
            timestamp=timestamp,
            flash_detected=is_flash,
            smoke_active=smoke_active,
            molly_active=molly_active,
            wall_active=wall_detected,
            recon_active=recon_active,
            ability_type=ability_type,
            screen_flash_intensity=flash_intensity,
            smoke_coverage_pct=smoke_coverage * 100,
            fire_coverage_pct=molly_coverage * 100,
        )
        self.frames.append(af)
        return af

    def generate_results(self) -> AbilityAnalysisResult:
        """Generate aggregated ability analysis results."""
        if not self.frames:
            return AbilityAnalysisResult(score=50.0)

        # Count by type
        flash_events = [e for e in self.events if e.ability_type == AbilityType.FLASH]
        smoke_events = [e for e in self.events if e.ability_type == AbilityType.SMOKE]
        molly_events = [e for e in self.events if e.ability_type == AbilityType.MOLLY]
        wall_events = [e for e in self.events if e.ability_type == AbilityType.WALL]
        recon_events = [e for e in self.events if e.ability_type == AbilityType.RECON]
        ult_events = [e for e in self.events if e.ability_type == AbilityType.ULTIMATE]

        # Average effectiveness per type
        flash_eff = (
            float(np.mean([e.effectiveness for e in flash_events]))
            if flash_events else 0.0
        )
        smoke_eff = (
            float(np.mean([e.effectiveness for e in smoke_events]))
            if smoke_events else 0.0
        )
        molly_eff = (
            float(np.mean([e.effectiveness for e in molly_events]))
            if molly_events else 0.0
        )

        total_abilities = len(self.events)
        overall_eff = (
            float(np.mean([e.effectiveness for e in self.events]))
            if self.events else 0.0
        )

        # Score calculation
        # Utility usage (having abilities): 40 points
        usage_score = min(40, total_abilities * 5)

        # Effectiveness of used abilities: 40 points
        eff_score = (overall_eff / 100) * 40

        # Variety of abilities used: 20 points
        types_used = len({
            e.ability_type for e in self.events
            if e.ability_type != AbilityType.UNKNOWN
        })
        variety_score = min(20, types_used * 5)

        total_score = usage_score + eff_score + variety_score

        # Build event dicts
        ability_events = [
            {
                "timestamp": e.timestamp,
                "type": e.ability_type,
                "duration": round(e.duration, 1),
                "effectiveness": round(e.effectiveness, 1),
                "description": e.description,
            }
            for e in self.events
        ]

        # Timeline (sampled)
        ability_timeline = [
            {
                "timestamp": f.timestamp,
                "flash": f.flash_detected,
                "smoke": f.smoke_active,
                "molly": f.molly_active,
                "wall": f.wall_active,
                "recon": f.recon_active,
                "type": f.ability_type,
            }
            for f in self.frames[::5]
        ]

        efficiency_breakdown = {
            "flash": {"count": len(flash_events), "avg_effectiveness": round(flash_eff, 1)},
            "smoke": {"count": len(smoke_events), "avg_effectiveness": round(smoke_eff, 1)},
            "molly": {"count": len(molly_events), "avg_effectiveness": round(molly_eff, 1)},
            "wall": {"count": len(wall_events), "avg_effectiveness": 60.0},
            "recon": {"count": len(recon_events), "avg_effectiveness": 70.0},
            "ultimate": {"count": len(ult_events), "avg_effectiveness": 80.0},
        }

        return AbilityAnalysisResult(
            total_abilities_used=total_abilities,
            flash_count=len(flash_events),
            smoke_count=len(smoke_events),
            molly_count=len(molly_events),
            wall_count=len(wall_events),
            recon_count=len(recon_events),
            ultimate_count=len(ult_events),
            flash_effectiveness=round(flash_eff, 1),
            smoke_effectiveness=round(smoke_eff, 1),
            molly_effectiveness=round(molly_eff, 1),
            overall_utility_score=round(overall_eff, 1),
            ability_events=ability_events,
            ability_timeline=ability_timeline,
            efficiency_breakdown=efficiency_breakdown,
            score=round(total_score, 1),
        )
