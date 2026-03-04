from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AnalysisCreate(BaseModel):
    filename: str


class TimelineEvent(BaseModel):
    timestamp: float
    event_type: str  # shot, kill, death, ability, round_start, round_end, callout
    description: str
    details: Optional[dict] = None


class CrosshairData(BaseModel):
    head_level_consistency: float  # % of time at head level
    avg_pre_aim_distance: float  # avg pixels from contact point
    first_contact_efficiency: float  # avg adjustment needed
    center_vs_edge_ratio: float  # % edge aiming (good)
    floor_aiming_percentage: float  # % time aiming at floor (bad)
    heatmap_points: list[dict]  # {x, y, count}


class MovementData(BaseModel):
    counter_strafe_accuracy: float  # % correct counter-strafes
    movement_while_shooting: float  # % of shots while moving (bad)
    peek_type_distribution: dict  # {tight: %, wide: %, over: %}
    spray_control_score: float


class DecisionData(BaseModel):
    multi_angle_exposure_count: int  # times exposed to >1 angle
    trade_efficiency: float  # % of deaths properly traded
    utility_impact_score: float
    commitment_clarity: float  # % clear fight/flee decisions


class CommunicationData(BaseModel):
    total_callouts: int
    timely_callouts: float  # % on-time
    late_callouts: float  # % late
    transcription_segments: list[dict]  # {start, end, text}


class Recommendation(BaseModel):
    priority: int  # 1-3
    category: str  # crosshair, movement, decision, communication
    title: str
    description: str
    practice_drill: Optional[str] = None


class ProComparison(BaseModel):
    player_name: str
    player_score: float
    nats_benchmark: float
    s0m_benchmark: float
    tenz_benchmark: float


class RoundAnalysis(BaseModel):
    round_number: int
    timestamp_start: float
    timestamp_end: float
    outcome: str  # win, loss
    crosshair_score: float
    movement_score: float
    decision_score: float
    key_moments: list[str]
    notes: str


class AnalysisResponse(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    total_frames_analyzed: Optional[int] = None
    overall_score: Optional[float] = None
    crosshair_score: Optional[float] = None
    movement_score: Optional[float] = None
    decision_score: Optional[float] = None
    communication_score: Optional[float] = None
    map_score: Optional[float] = None
    status_text: Optional[str] = None
    crosshair_data: Optional[dict] = None
    movement_data: Optional[dict] = None
    decision_data: Optional[dict] = None
    communication_data: Optional[dict] = None
    timeline_events: Optional[list] = None
    recommendations: Optional[list] = None
    heatmap_data: Optional[dict] = None
    round_analysis: Optional[list] = None
    pro_comparison: Optional[dict] = None
    map_data: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AnalysisListItem(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    overall_score: Optional[float] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    id: str
    message: str
