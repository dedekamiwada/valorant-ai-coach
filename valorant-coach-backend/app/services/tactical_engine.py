"""
Tactical Decision Engine for Valorant VODs — the CORE of the system.

Analyses all game state data and produces specific tactical recommendations
in the required output format:

    "No timestamp HH:MM:SS → Você deveria ter [ação específica]
     porque [razão tática detalhada]"

Decision categories:
- ROTAÇÃO: When to rotate between sites
- HOLD: When to hold position
- RETAKE: When and how to retake a site
- EXECUTE: When to execute a site take
- LURK: When to play lurk
- SAVE: When to save weapons/economy
- ABILITY USAGE: When/where to use specific abilities
- COMBAT: Mechanical advice (pre-aim, peek type, counter-strafe)
- ECONOMY: Buy/save decisions
"""

import numpy as np
from dataclasses import dataclass, field


def _fmt_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class TacticalRecommendation:
    """A single tactical recommendation tied to a specific timestamp."""
    timestamp: float  # seconds into the VOD
    timestamp_fmt: str = ""  # HH:MM:SS
    action: str = ""  # what they should have done
    reason: str = ""  # tactical reasoning
    category: str = ""  # rotate, hold, retake, execute, lurk, save, ability, combat, economy
    priority: int = 2  # 1=critical, 2=important, 3=minor
    confidence: float = 0.5

    # Full formatted recommendation
    formatted: str = ""

    def __post_init__(self):
        if not self.timestamp_fmt:
            self.timestamp_fmt = _fmt_timestamp(self.timestamp)
        if not self.formatted and self.action and self.reason:
            self.formatted = (
                f"No timestamp {self.timestamp_fmt} → "
                f"Você deveria ter {self.action} porque {self.reason}"
            )


@dataclass
class TacticalAnalysisResult:
    """Complete tactical analysis output."""
    recommendations: list[dict] = field(default_factory=list)
    total_recommendations: int = 0
    critical_count: int = 0
    important_count: int = 0
    minor_count: int = 0
    categories_breakdown: dict = field(default_factory=dict)
    score: float = 0.0


class TacticalEngine:
    """
    The core decision engine that synthesises data from all analysers
    to produce actionable, timestamp-specific tactical recommendations.

    Input data comes from:
    - GameStateParser: alive/dead, round phase, spike, economy
    - CrosshairAnalyzer: aim quality per frame
    - MovementAnalyzer: movement quality per frame
    - DecisionAnalyzer: exposure, utility usage
    - AbilityAnalyzer: ability events and efficiency
    - MapAnalyzer: positioning, zone, rotation
    - AudioProcessor: communication events
    """

    # Thresholds for generating recommendations
    MULTI_ANGLE_THRESHOLD = 2  # exposed to >2 angles = bad
    LOW_HP_TEAMMATES_THRESHOLD = 2  # fewer than 2 allies = consider saving
    SAVE_DISADVANTAGE = 3  # 1vN where N >= 3 → save
    EXPOSED_ZONE_TIME = 5.0  # seconds in exposed position → warn
    ROTATION_TOO_SLOW = 6.0  # seconds to rotate → too slow
    FLOOR_AIM_CONSECUTIVE = 3  # consecutive floor-aim frames → warn

    def __init__(self):
        self.recommendations: list[TacticalRecommendation] = []

    def analyse_rotation_decisions(
        self,
        game_states: list[dict],
        map_frames: list[dict],
        zone_changes: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate rotation-related recommendations."""
        recs: list[TacticalRecommendation] = []

        for zc in zone_changes:
            ts = zc.get("timestamp", 0)
            from_zone = zc.get("from_zone", "unknown")
            to_zone = zc.get("to_zone", "unknown")
            duration = zc.get("duration", 0)

            # Slow rotation
            if duration > self.ROTATION_TOO_SLOW:
                recs.append(TacticalRecommendation(
                    timestamp=ts,
                    action=f"rotacionado mais rápido de {from_zone} para {to_zone}",
                    reason=(
                        f"a rotação levou {duration:.1f}s, o que é muito lento. "
                        f"Rotações eficientes devem levar menos de {self.ROTATION_TOO_SLOW:.0f}s. "
                        "Use atalhos no mapa e corra com a faca equipada para chegar mais rápido."
                    ),
                    category="rotate",
                    priority=2,
                    confidence=0.7,
                ))

        # Unnecessary rotations (rotating back to same zone quickly)
        for i in range(1, len(zone_changes)):
            curr = zone_changes[i]
            prev = zone_changes[i - 1]
            if (
                curr.get("to_zone") == prev.get("from_zone")
                and curr["timestamp"] - prev["timestamp"] < 10
            ):
                recs.append(TacticalRecommendation(
                    timestamp=curr["timestamp"],
                    action=f"ficado em {prev.get('from_zone', 'sua posição')} ao invés de rotacionar desnecessariamente",
                    reason=(
                        "você rotacionou e voltou para a posição original em menos de 10 segundos. "
                        "Isso desperdiça tempo e expõe você durante o trânsito. "
                        "Comprometa-se com a rotação ou fique na posição original."
                    ),
                    category="rotate",
                    priority=2,
                    confidence=0.8,
                ))

        return recs

    def analyse_economy_decisions(
        self,
        game_states: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate economy-related recommendations."""
        recs: list[TacticalRecommendation] = []

        for gs in game_states:
            ts = gs.get("timestamp", 0)
            buy_type = gs.get("economy", {}).get("buy_type", "unknown")
            credits = gs.get("economy", {}).get("player_credits", 0)
            round_num = gs.get("round_number", 0)
            ally_score = gs.get("ally_score", 0)
            enemy_score = gs.get("enemy_score", 0)

            # Force buy when should eco
            if buy_type == "force_buy" and credits < 2000:
                loss_streak = enemy_score - ally_score
                if loss_streak >= 2:
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="economizado ao invés de forçar compra",
                        reason=(
                            f"com apenas {credits} créditos e {loss_streak} rounds perdidos consecutivos, "
                            "forçar compra desperdiça dinheiro. "
                            "Economize para um full buy no próximo round — "
                            "o bônus de derrota vai garantir dinheiro suficiente."
                        ),
                        category="economy",
                        priority=1,
                        confidence=0.75,
                    ))

            # Eco when team has money
            if buy_type == "eco" and credits >= 3900 and round_num > 2:
                recs.append(TacticalRecommendation(
                    timestamp=ts,
                    action="comprado armas completas",
                    reason=(
                        f"você tinha {credits} créditos mas fez eco. "
                        "Com dinheiro suficiente para rifle + armadura + habilidades, "
                        "não há razão para economizar. Full buy maximiza suas chances de ganhar o round."
                    ),
                    category="economy",
                    priority=1,
                    confidence=0.8,
                ))

        return recs

    def analyse_positioning_decisions(
        self,
        map_frames: list[dict],
        game_states: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate positioning-related recommendations."""
        recs: list[TacticalRecommendation] = []

        exposed_start: float | None = None

        for mf in map_frames:
            ts = mf.get("timestamp", 0)
            is_exposed = mf.get("exposed", mf.get("is_exposed", False))
            zone = mf.get("zone", "unknown")

            if is_exposed:
                if exposed_start is None:
                    exposed_start = ts
                elif ts - exposed_start > self.EXPOSED_ZONE_TIME:
                    recs.append(TacticalRecommendation(
                        timestamp=exposed_start,
                        action=f"recuado para uma posição com cobertura em {zone}",
                        reason=(
                            f"você ficou exposto sem suporte do time por {ts - exposed_start:.1f}s. "
                            "Posições agressivas sem cobertura de companheiros são extremamente perigosas. "
                            "Recue para uma posição onde pelo menos um aliado possa trocar sua morte."
                        ),
                        category="hold",
                        priority=1,
                        confidence=0.7,
                    ))
                    exposed_start = None
            else:
                exposed_start = None

        return recs

    def analyse_combat_decisions(
        self,
        crosshair_frames: list[dict],
        movement_frames: list[dict],
        decision_frames: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate combat mechanic recommendations."""
        recs: list[TacticalRecommendation] = []

        # Detect prolonged floor aiming
        floor_aim_count = 0
        floor_aim_start: float | None = None
        for cf in crosshair_frames:
            ts = cf.get("timestamp", 0)
            if cf.get("floor_aiming", False):
                floor_aim_count += 1
                if floor_aim_start is None:
                    floor_aim_start = ts
            else:
                if floor_aim_count >= self.FLOOR_AIM_CONSECUTIVE and floor_aim_start is not None:
                    recs.append(TacticalRecommendation(
                        timestamp=floor_aim_start,
                        action="mantido a mira na altura da cabeça durante a movimentação",
                        reason=(
                            "sua mira caiu para o chão por vários segundos consecutivos. "
                            "Mesmo durante rotações, a mira deve estar sempre na altura da cabeça. "
                            "Isso reduz drasticamente o tempo de reação ao encontrar um inimigo."
                        ),
                        category="combat",
                        priority=1,
                        confidence=0.85,
                    ))
                floor_aim_count = 0
                floor_aim_start = None

        # Detect shooting while moving
        for mf in movement_frames:
            ts = mf.get("timestamp", 0)
            if mf.get("moving", False) and mf.get("shooting", False):
                if not mf.get("counter_strafe", False):
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="feito counter-strafe antes de atirar",
                        reason=(
                            "você atirou enquanto se movia sem fazer counter-strafe. "
                            "Em Valorant, a precisão cai drasticamente em movimento. "
                            "Pressione a tecla oposta (A→D ou D→A) para parar instantaneamente antes de atirar."
                        ),
                        category="combat",
                        priority=1,
                        confidence=0.9,
                    ))

        # Detect multi-angle exposure
        for df in decision_frames:
            ts = df.get("timestamp", 0)
            angles = df.get("angles", df.get("exposed_angles", 1))
            cover = df.get("cover", df.get("is_using_cover", True))
            if angles >= 3 and not cover:
                recs.append(TacticalRecommendation(
                    timestamp=ts,
                    action="usado smoke ou flash antes de se expor a múltiplos ângulos",
                    reason=(
                        f"você ficou exposto a {angles} ângulos simultâneos sem usar utilitários. "
                        "Nunca entre em uma posição onde múltiplos inimigos podem te ver ao mesmo tempo. "
                        "Use smoke para bloquear um ângulo e lide com um inimigo de cada vez."
                    ),
                    category="ability",
                    priority=1,
                    confidence=0.75,
                ))

        return recs

    def analyse_save_decisions(
        self,
        game_states: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate save/retake decisions based on player advantage."""
        recs: list[TacticalRecommendation] = []

        for gs in game_states:
            ts = gs.get("timestamp", 0)
            allies = gs.get("allies_alive", 5)
            enemies = gs.get("enemies_alive", 5)
            phase = gs.get("round_phase", "play")

            # 1vN where N >= 3 and no spike planted = save
            if (
                allies == 1
                and enemies >= self.SAVE_DISADVANTAGE
                and phase == "play"
            ):
                spike_planted = gs.get("spike", {}).get("is_planted", False)
                if not spike_planted:
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="salvado sua arma ao invés de tentar clutch",
                        reason=(
                            f"com situação de 1v{enemies} sem spike plantada, "
                            "a chance de ganhar o round é mínima. "
                            "Salvar sua arma garante economia melhor para o próximo round "
                            "e dá uma chance real de vencer."
                        ),
                        category="save",
                        priority=1,
                        confidence=0.85,
                    ))

            # Retake opportunity
            if (
                allies >= 3
                and enemies <= 2
                and phase in ("post_plant", "POST_PLANT")
            ):
                recs.append(TacticalRecommendation(
                    timestamp=ts,
                    action=f"agrupado com o time para retake ({allies}v{enemies})",
                    reason=(
                        f"com vantagem numérica de {allies}v{enemies} e spike plantada, "
                        "vocês têm uma excelente oportunidade de retake. "
                        "Agrupem-se, usem utilitários juntos e entrem coordenados no site."
                    ),
                    category="retake",
                    priority=2,
                    confidence=0.7,
                ))

        return recs

    def analyse_ability_usage(
        self,
        ability_events: list[dict],
        game_states: list[dict],
    ) -> list[TacticalRecommendation]:
        """Generate ability usage recommendations."""
        recs: list[TacticalRecommendation] = []

        for ae in ability_events:
            ts = ae.get("timestamp", 0)
            ab_type = ae.get("type", "unknown")
            effectiveness = ae.get("effectiveness", 50)

            # Low effectiveness abilities
            if effectiveness < 30:
                if ab_type == "flash":
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="esperado o timing certo para usar a flash",
                        reason=(
                            "a flash não cegou inimigos efetivamente. "
                            "Use flashes em pop-flashes (flash que estoura ao sair da parede) "
                            "ou coordene com seu time para entrar imediatamente após a flash."
                        ),
                        category="ability",
                        priority=2,
                        confidence=0.6,
                    ))
                elif ab_type == "smoke":
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="posicionado a smoke para bloquear uma linha de visão importante",
                        reason=(
                            "a smoke não cobriu uma posição estratégica efetiva. "
                            "Smokes devem bloquear ângulos específicos que os inimigos usam, "
                            "como entradas de site, heaven, ou posições de AWP."
                        ),
                        category="ability",
                        priority=2,
                        confidence=0.6,
                    ))
                elif ab_type == "molly":
                    recs.append(TacticalRecommendation(
                        timestamp=ts,
                        action="guardado a molly para o pós-plant ou denial de entrada",
                        reason=(
                            "a molly não teve impacto significativo. "
                            "Mollies são mais valiosas no pós-plant (forçar o defuse a parar) "
                            "ou para negar entrada em posições específicas."
                        ),
                        category="ability",
                        priority=2,
                        confidence=0.6,
                    ))

        return recs

    def generate_recommendations(
        self,
        game_states: list[dict] | None = None,
        crosshair_frames: list[dict] | None = None,
        movement_frames: list[dict] | None = None,
        decision_frames: list[dict] | None = None,
        map_frames: list[dict] | None = None,
        zone_changes: list[dict] | None = None,
        ability_events: list[dict] | None = None,
    ) -> TacticalAnalysisResult:
        """
        Run the full tactical analysis and produce timestamped recommendations.

        Aggregates insights from all sub-analysers.
        """
        all_recs: list[TacticalRecommendation] = []

        # Rotation decisions
        if zone_changes:
            all_recs.extend(
                self.analyse_rotation_decisions(
                    game_states or [], map_frames or [], zone_changes
                )
            )

        # Economy decisions
        if game_states:
            all_recs.extend(self.analyse_economy_decisions(game_states))

        # Positioning decisions
        if map_frames:
            all_recs.extend(
                self.analyse_positioning_decisions(
                    map_frames, game_states or []
                )
            )

        # Combat decisions
        if crosshair_frames or movement_frames or decision_frames:
            all_recs.extend(
                self.analyse_combat_decisions(
                    crosshair_frames or [],
                    movement_frames or [],
                    decision_frames or [],
                )
            )

        # Save/retake decisions
        if game_states:
            all_recs.extend(self.analyse_save_decisions(game_states))

        # Ability usage decisions
        if ability_events:
            all_recs.extend(
                self.analyse_ability_usage(ability_events, game_states or [])
            )

        # Deduplicate: no two recommendations within 2 seconds of same category
        deduped: list[TacticalRecommendation] = []
        for rec in sorted(all_recs, key=lambda r: (r.priority, r.timestamp)):
            dominated = False
            for existing in deduped:
                if (
                    existing.category == rec.category
                    and abs(existing.timestamp - rec.timestamp) < 2.0
                ):
                    dominated = True
                    break
            if not dominated:
                deduped.append(rec)

        # Sort by timestamp
        deduped.sort(key=lambda r: r.timestamp)

        # Limit to top recommendations (prioritise critical ones)
        max_recs = 25
        if len(deduped) > max_recs:
            critical = [r for r in deduped if r.priority == 1]
            important = [r for r in deduped if r.priority == 2]
            minor = [r for r in deduped if r.priority == 3]
            selected = critical[:15] + important[:8] + minor[:2]
            selected.sort(key=lambda r: r.timestamp)
            deduped = selected[:max_recs]

        self.recommendations = deduped

        # Build output
        rec_dicts = [
            {
                "timestamp": r.timestamp,
                "timestamp_fmt": r.timestamp_fmt,
                "action": r.action,
                "reason": r.reason,
                "category": r.category,
                "priority": r.priority,
                "confidence": round(r.confidence, 2),
                "formatted": r.formatted,
            }
            for r in deduped
        ]

        # Category breakdown
        categories: dict[str, int] = {}
        for r in deduped:
            categories[r.category] = categories.get(r.category, 0) + 1

        critical_count = sum(1 for r in deduped if r.priority == 1)
        important_count = sum(1 for r in deduped if r.priority == 2)
        minor_count = sum(1 for r in deduped if r.priority == 3)

        # Score: fewer critical issues = higher score
        # Max 100, lose 5 for each critical, 2 for each important
        score = max(0, 100 - critical_count * 5 - important_count * 2 - minor_count)

        return TacticalAnalysisResult(
            recommendations=rec_dicts,
            total_recommendations=len(deduped),
            critical_count=critical_count,
            important_count=important_count,
            minor_count=minor_count,
            categories_breakdown=categories,
            score=round(score, 1),
        )
