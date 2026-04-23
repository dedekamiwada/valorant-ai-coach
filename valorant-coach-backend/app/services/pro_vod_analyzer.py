"""
Pro VOD Analyzer.

Converts a :class:`PipelineResult` (produced by ``video_pipeline.process_video``)
into a list of textual "strengths" describing what a pro player does well in
their VOD.

The existing analysis pipeline focuses on **weaknesses** (recommendations for
improvement).  When the uploaded video belongs to a pro player we want the
opposite: concrete, written-down observations about *what makes the pro so
good* that can be persisted in the knowledge base and consulted later as
reference material.

Each strength is returned as a dict ready to be persisted as a
``KnowledgeEntry`` row with ``source_type="pro_vod"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.video_pipeline import PipelineResult


@dataclass
class ProMetadata:
    """Metadata about the pro player whose VOD is being analysed."""

    dataset_id: str
    player_name: str | None
    team: str | None
    agent: str | None
    map_name: str | None
    rank: str | None = None


# Score thresholds for "this is a strength worth writing down" per category.
# Values are calibrated so that any above-average pro performance (typical
# benchmarks are 85-95 depending on the category) triggers at least one
# entry per category while mediocre numbers do not pollute the knowledge base.
STRENGTH_THRESHOLDS = {
    "crosshair": 70.0,
    "movement": 70.0,
    "decision": 70.0,
    "communication": 70.0,
    "map": 70.0,
    "overall": 75.0,
}


def _confidence_for(score: float) -> float:
    """Map a 0-100 score to a 0-1 confidence value."""
    return round(max(0.0, min(1.0, score / 100.0)), 3)


def _tag_list(meta: ProMetadata) -> list[str]:
    tags = ["pro_vod", "strength"]
    if meta.player_name:
        tags.append(meta.player_name)
    if meta.agent:
        tags.append(meta.agent)
    if meta.map_name:
        tags.append(meta.map_name)
    return tags


def _entry(
    meta: ProMetadata,
    category: str,
    subcategory: str | None,
    title: str,
    description: str,
    metric_name: str | None = None,
    metric_value: float | None = None,
    confidence: float = 0.7,
) -> dict[str, Any]:
    """Build a KnowledgeEntry-shaped dict for a pro strength."""
    return {
        "source_type": "pro_vod",
        "source_id": meta.dataset_id,
        "category": category,
        "subcategory": subcategory,
        "agent": meta.agent,
        "map_name": meta.map_name,
        "rank": meta.rank,
        "title": title,
        "description": description,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "confidence": confidence,
        "tags": _tag_list(meta),
    }


def _pro_ref(meta: ProMetadata) -> str:
    """Short reference like 'nAts (Cypher · Haven)' for use in titles/descriptions."""
    bits: list[str] = []
    if meta.player_name:
        bits.append(meta.player_name)
    ctx: list[str] = []
    if meta.agent:
        ctx.append(meta.agent)
    if meta.map_name:
        ctx.append(meta.map_name)
    ref = " ".join(bits) if bits else "O pro"
    if ctx:
        ref += f" ({' · '.join(ctx)})"
    return ref


# ── Category extractors ──────────────────────────────────────────────

def _crosshair_strengths(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    score = result.crosshair_score
    if score < STRENGTH_THRESHOLDS["crosshair"]:
        return []

    data = result.crosshair_data or {}
    head = float(data.get("head_level_consistency", 0.0))
    floor = float(data.get("floor_aiming_percentage", 0.0))
    edge = float(data.get("center_vs_edge_ratio", 0.0))
    pre_aim = float(data.get("avg_pre_aim_distance", 0.0))
    pro = _pro_ref(meta)
    entries: list[dict] = []

    entries.append(_entry(
        meta,
        category="crosshair",
        subcategory="overall",
        title=f"{pro} · Crosshair placement de elite ({score:.0f}/100)",
        description=(
            f"{pro} obteve {score:.0f}/100 em crosshair placement na análise deste VOD. "
            f"Mantém a mira na altura da cabeça {head:.0f}% do tempo, mira no chão apenas {floor:.0f}%, "
            f"e prioriza as bordas das aberturas em {edge:.0f}% das situações. "
            "Essa consistência mecânica é o que permite a ele ganhar duelos reagindo mais rápido "
            "que o inimigo e não sendo forçado a grandes ajustes verticais no primeiro contato."
        ),
        metric_name="crosshair_score",
        metric_value=float(score),
        confidence=_confidence_for(score),
    ))

    if head >= 80.0:
        entries.append(_entry(
            meta,
            category="crosshair",
            subcategory="head_level",
            title=f"{pro} · Mira sempre na altura da cabeça ({head:.0f}%)",
            description=(
                f"{pro} mantém a mira na altura da cabeça em {head:.0f}% dos frames analisados. "
                "Ele ajusta a mira preventivamente ao cruzar cada porta, canto e transição de posição, "
                "de forma que nunca precisa puxar o mouse para cima para encontrar uma cabeça. "
                "Esse hábito praticamente elimina headshots perdidos por ajuste vertical."
            ),
            metric_name="head_level_consistency",
            metric_value=head,
            confidence=_confidence_for(head),
        ))

    if edge >= 55.0:
        entries.append(_entry(
            meta,
            category="crosshair",
            subcategory="edge_aiming",
            title=f"{pro} · Pré-aim nas bordas das aberturas ({edge:.0f}%)",
            description=(
                f"Em {edge:.0f}% das situações {pro} posiciona a mira na borda onde o inimigo "
                "aparece primeiro, em vez do centro do corredor/porta. Isso reduz drasticamente "
                "o tempo de reação necessário: quando a cabeça do inimigo surge, a mira já está sobre o pixel certo."
            ),
            metric_name="center_vs_edge_ratio",
            metric_value=edge,
            confidence=_confidence_for(edge),
        ))

    if pre_aim > 0 and pre_aim <= 5.0:
        entries.append(_entry(
            meta,
            category="crosshair",
            subcategory="pre_aim",
            title=f"{pro} · Ajuste de pré-aim mínimo ({pre_aim:.1f}px médio)",
            description=(
                f"Distância média de ajuste entre a posição pré-contato da mira e o ponto real "
                f"do primeiro tiro: {pre_aim:.1f}px. Um número tão baixo só é possível quando o "
                "pré-aim em cada canto é estudado — ele não 'procura' o inimigo, apenas confirma."
            ),
            metric_name="avg_pre_aim_distance",
            metric_value=pre_aim,
            confidence=0.75,
        ))

    return entries


def _movement_strengths(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    score = result.movement_score
    if score < STRENGTH_THRESHOLDS["movement"]:
        return []

    data = result.movement_data or {}
    counter = float(data.get("counter_strafe_accuracy", 0.0))
    move_shoot = float(data.get("movement_while_shooting", 100.0))
    spray = float(data.get("spray_control_score", 0.0))
    pro = _pro_ref(meta)
    entries: list[dict] = []

    entries.append(_entry(
        meta,
        category="movement",
        subcategory="overall",
        title=f"{pro} · Controle de movimento profissional ({score:.0f}/100)",
        description=(
            f"Score de movimento: {score:.0f}/100. Counter-strafe correto em {counter:.0f}% dos tiros, "
            f"atira em movimento em apenas {move_shoot:.0f}%, controle de spray {spray:.0f}/100. "
            "O movimento dele nunca é aleatório — cada passo é feito com a intenção de conseguir "
            "parar limpo antes do próximo tiro."
        ),
        metric_name="movement_score",
        metric_value=float(score),
        confidence=_confidence_for(score),
    ))

    if counter >= 80.0:
        entries.append(_entry(
            meta,
            category="movement",
            subcategory="counter_strafe",
            title=f"{pro} · Counter-strafe consistente ({counter:.0f}%)",
            description=(
                f"{pro} executa counter-strafe corretamente em {counter:.0f}% dos engajamentos: "
                "pressiona a tecla contrária antes de atirar, zerando a inércia e recuperando a "
                "precisão total em ~80ms. Essa é a mecânica base que separa tiros 'a moleque' de "
                "tiros de rifle precisos."
            ),
            metric_name="counter_strafe_accuracy",
            metric_value=counter,
            confidence=_confidence_for(counter),
        ))

    if move_shoot <= 15.0:
        entries.append(_entry(
            meta,
            category="movement",
            subcategory="stop_before_shooting",
            title=f"{pro} · Praticamente não atira em movimento ({move_shoot:.0f}%)",
            description=(
                f"Apenas {move_shoot:.0f}% dos tiros de {pro} acontecem com o personagem em movimento. "
                "A norma dele é parar antes de pressionar o gatilho — um hábito que em Valorant vale "
                "dezenas de por cento em accuracy real."
            ),
            metric_name="movement_while_shooting",
            metric_value=move_shoot,
            confidence=_confidence_for(100.0 - move_shoot),
        ))

    return entries


def _decision_strengths(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    score = result.decision_score
    if score < STRENGTH_THRESHOLDS["decision"]:
        return []

    data = result.decision_data or {}
    multi_angle = int(data.get("multi_angle_exposure_count", 999))
    trade = float(data.get("trade_efficiency", 0.0))
    utility = float(data.get("utility_impact_score", 0.0))
    commit = float(data.get("commitment_clarity", 0.0))
    pro = _pro_ref(meta)
    entries: list[dict] = []

    entries.append(_entry(
        meta,
        category="decision",
        subcategory="overall",
        title=f"{pro} · Tomada de decisão de alto nível ({score:.0f}/100)",
        description=(
            f"Score de decisão: {score:.0f}/100. Exposição a múltiplos ângulos: {multi_angle} vezes, "
            f"trade efficiency {trade:.0f}%, impacto de utility {utility:.0f}/100, "
            f"clareza de commitment {commit:.0f}%. {pro} escolhe os duelos: ou pica com vantagem "
            "de utility/informação, ou recua sem tomar dano."
        ),
        metric_name="decision_score",
        metric_value=float(score),
        confidence=_confidence_for(score),
    ))

    if multi_angle <= 5:
        entries.append(_entry(
            meta,
            category="decision",
            subcategory="angle_isolation",
            title=f"{pro} · Isola duelos — só {multi_angle} exposições a múltiplos ângulos",
            description=(
                f"Em toda a VOD, {pro} se expôs a mais de um ângulo em apenas {multi_angle} "
                "ocasiões. Ele prefere usar smoke/flash para cortar linhas de visão ou reposicionar "
                "antes de picar, mantendo todo engajamento como 1v1."
            ),
            metric_name="multi_angle_exposure_count",
            metric_value=float(multi_angle),
            confidence=0.8,
        ))

    if trade >= 70.0:
        entries.append(_entry(
            meta,
            category="decision",
            subcategory="trade_efficiency",
            title=f"{pro} · Trade efficiency {trade:.0f}%",
            description=(
                f"{trade:.0f}% das mortes no time de {pro} são trocadas. Isso indica rotação e "
                "timing de entry/suporte afinados: o segundo jogador do time já está com mira no "
                "ângulo no momento em que o primeiro morre."
            ),
            metric_name="trade_efficiency",
            metric_value=trade,
            confidence=_confidence_for(trade),
        ))

    if utility >= 70.0:
        entries.append(_entry(
            meta,
            category="decision",
            subcategory="utility_usage",
            title=f"{pro} · Uso de utility com impacto ({utility:.0f}/100)",
            description=(
                f"Cada smoke, flash e molly de {pro} tem impacto mensurável (score {utility:.0f}/100). "
                "Utility raramente é usado 'por usar' — sempre tem um objetivo claro: cortar visão "
                "pra uma entry, cegar um ângulo específico ou atrasar uma rotação."
            ),
            metric_name="utility_impact_score",
            metric_value=utility,
            confidence=_confidence_for(utility),
        ))

    return entries


def _communication_strengths(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    score = result.communication_score
    if score < STRENGTH_THRESHOLDS["communication"]:
        return []

    data = result.communication_data or {}
    total = int(data.get("total_callouts", 0))
    timely = float(data.get("timely_callouts_pct", data.get("timely_callouts", 0.0)))
    pro = _pro_ref(meta)

    return [_entry(
        meta,
        category="communication",
        subcategory="callouts",
        title=f"{pro} · Callouts no tempo certo ({timely:.0f}% timely)",
        description=(
            f"{total} callouts no total, {timely:.0f}% dentro da janela em que a informação "
            f"ainda é acionável. Informação útil é informação rápida: {pro} reporta o que importa "
            "(número, HP, utility usada) em 1-2 segundos após o contato."
        ),
        metric_name="timely_callouts_pct",
        metric_value=timely,
        confidence=_confidence_for(score),
    )]


def _map_strengths(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    score = result.map_score
    if score < STRENGTH_THRESHOLDS["map"]:
        return []

    data = result.map_data or {}
    positioning = float(data.get("positioning_score", score))
    exposed = float(data.get("exposed_positioning_pct", 100.0))
    rot_time = float(data.get("avg_rotation_time", 0.0))
    pro = _pro_ref(meta)
    entries: list[dict] = []

    entries.append(_entry(
        meta,
        category="positioning",
        subcategory="overall",
        title=f"{pro} · Posicionamento e leitura de mapa ({score:.0f}/100)",
        description=(
            f"Score de mapa: {score:.0f}/100, posicionamento {positioning:.0f}/100. "
            f"Exposto em apenas {exposed:.0f}% do tempo, rotação média em {rot_time:.1f}s. "
            f"{pro} ocupa posições que permitem recuar ou trocar sem ficar em crossfire desfavorável."
        ),
        metric_name="map_score",
        metric_value=float(score),
        confidence=_confidence_for(score),
    ))

    if exposed <= 25.0:
        entries.append(_entry(
            meta,
            category="positioning",
            subcategory="cover_usage",
            title=f"{pro} · Exposição mínima ({exposed:.0f}%)",
            description=(
                f"Apenas {exposed:.0f}% do tempo em posições expostas. O padrão é: encostar no "
                "ângulo, fazer a informação e recuar para cobertura. Nunca fica parado no meio de "
                "uma linha de crossfire."
            ),
            metric_name="exposed_positioning_pct",
            metric_value=exposed,
            confidence=_confidence_for(100.0 - exposed),
        ))

    return entries


def _overall_strength(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    if result.overall_score < STRENGTH_THRESHOLDS["overall"]:
        return []
    pro = _pro_ref(meta)
    return [_entry(
        meta,
        category="overall",
        subcategory=None,
        title=f"{pro} · Referência geral ({result.overall_score:.0f}/100)",
        description=(
            f"Score geral de {result.overall_score:.0f}/100 nesta VOD. "
            f"Crosshair {result.crosshair_score:.0f}, movimento {result.movement_score:.0f}, "
            f"decisão {result.decision_score:.0f}, comunicação {result.communication_score:.0f}, "
            f"mapa {result.map_score:.0f}. Use este VOD como referência ao treinar "
            f"{meta.agent or 'este agente'} em {meta.map_name or 'este mapa'}."
        ),
        metric_name="overall_score",
        metric_value=float(result.overall_score),
        confidence=_confidence_for(result.overall_score),
    )]


def _highlight_moments(meta: ProMetadata, result: PipelineResult) -> list[dict]:
    """Pick a couple of notable timeline events (kills / multi-kills) as
    concrete moments to study."""
    events = result.timeline_events or []
    kill_events = [e for e in events if e.get("event_type") == "kill"]
    if not kill_events:
        return []
    pro = _pro_ref(meta)

    # Take up to 3 kill moments spaced out across the VOD
    step = max(1, len(kill_events) // 3)
    picked = kill_events[::step][:3]
    entries: list[dict] = []
    for idx, ev in enumerate(picked, start=1):
        ts = float(ev.get("timestamp", 0.0))
        desc = ev.get("description", "kill")
        entries.append(_entry(
            meta,
            category="highlight",
            subcategory="kill_moment",
            title=f"{pro} · Momento-chave #{idx} aos {_fmt_ts(ts)}",
            description=(
                f"Aos {_fmt_ts(ts)} da VOD: {desc}. Assista com atenção ao crosshair placement, "
                "uso de utility imediatamente antes, e decisão de peek/hold no momento do kill."
            ),
            metric_name="timestamp_seconds",
            metric_value=ts,
            confidence=0.6,
        ))
    return entries


def _fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


# ── Public API ───────────────────────────────────────────────────────

def extract_pro_strengths(
    pipeline_result: PipelineResult,
    meta: ProMetadata,
) -> list[dict[str, Any]]:
    """Extract 'what the pro does well' entries from a PipelineResult.

    Returns a list of dicts, each one shaped to become a ``KnowledgeEntry``
    row (``source_type="pro_vod"``, ``source_id=<dataset_id>``).
    """
    entries: list[dict[str, Any]] = []
    entries.extend(_overall_strength(meta, pipeline_result))
    entries.extend(_crosshair_strengths(meta, pipeline_result))
    entries.extend(_movement_strengths(meta, pipeline_result))
    entries.extend(_decision_strengths(meta, pipeline_result))
    entries.extend(_communication_strengths(meta, pipeline_result))
    entries.extend(_map_strengths(meta, pipeline_result))
    entries.extend(_highlight_moments(meta, pipeline_result))
    return entries


def summarize_pro_strengths(entries: list[dict[str, Any]]) -> str:
    """Human-readable one-paragraph summary of the generated strengths,
    used as the top-level report text returned to the UI after analysis."""
    if not entries:
        return (
            "Nenhum ponto forte acima do threshold foi detectado nesta VOD. "
            "Tente um clip mais longo ou verifique se o vídeo é de gameplay real."
        )
    by_cat: dict[str, int] = {}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    parts = [f"{n} em {cat}" for cat, n in sorted(by_cat.items())]
    return (
        f"{len(entries)} pontos fortes extraídos do VOD do pro — "
        + ", ".join(parts)
        + "."
    )
