export interface AnalysisListItem {
  id: string;
  filename: string;
  status: string;
  progress: number;
  status_text?: string | null;
  overall_score: number | null;
  created_at: string | null;
}

export interface TimelineEvent {
  timestamp: number;
  event_type: string;
  description: string;
}

export interface CrosshairFrameData {
  timestamp: number;
  head_level: boolean;
  floor_aiming: boolean;
  edge_aiming: boolean;
  combat: boolean;
  adjustment: number;
}

export interface MovementFrameData {
  timestamp: number;
  moving: boolean;
  shooting: boolean;
  magnitude: number;
  peek: string;
  counter_strafe: boolean;
}

export interface HeatmapPoint {
  x: number;
  y: number;
  value: number;
}

export interface CrosshairData {
  head_level_consistency: number;
  avg_pre_aim_distance: number;
  first_contact_efficiency: number;
  center_vs_edge_ratio: number;
  floor_aiming_percentage: number;
  heatmap_points: HeatmapPoint[];
  frame_data: CrosshairFrameData[];
}

export interface MovementData {
  counter_strafe_accuracy: number;
  movement_while_shooting: number;
  peek_type_distribution: Record<string, number>;
  spray_control_score: number;
  frame_data: MovementFrameData[];
}

export interface DecisionData {
  multi_angle_exposure_count: number;
  trade_efficiency: number;
  utility_impact_score: number;
  commitment_clarity: number;
  exposure_timeline: { timestamp: number; angles: number; cover: boolean }[];
  utility_events: { timestamp: number; type: string }[];
}

export interface CommunicationData {
  total_callouts: number;
  timely_callouts_pct: number;
  late_callouts_pct: number;
  transcription_segments: {
    start: number;
    end: number;
    text: string;
    is_callout: boolean;
    is_timely: boolean;
  }[];
  audio_events: unknown[];
}

export interface PositioningEvent {
  timestamp: number;
  event_type: string;
  description: string;
  severity: string;
}

export interface ZoneTimeline {
  timestamp: number;
  zone: string;
  duration: number;
}

export interface MapData {
  positioning_score: number;
  time_in_zones: Record<string, number>;
  rotation_count: number;
  avg_rotation_time: number;
  exposed_positioning_pct: number;
  zone_timeline: ZoneTimeline[];
  positioning_events: PositioningEvent[];
}

export interface Recommendation {
  priority: number;
  category: string;
  title: string;
  description: string;
  practice_drill?: string;
}

export interface RoundAnalysis {
  round_number: number;
  timestamp_start: number;
  timestamp_end: number;
  outcome: string;
  crosshair_score: number;
  movement_score: number;
  decision_score: number;
  key_moments: string[];
  notes: string;
}

export interface ProBenchmarks {
  crosshair: number;
  movement: number;
  decision: number;
  communication: number;
  map?: number;
  overall: number;
}

export interface ProComparison {
  player_overall: number;
  player_crosshair: number;
  player_movement: number;
  player_decision: number;
  player_communication: number;
  player_map?: number;
  benchmarks: Record<string, ProBenchmarks>;
}

export interface HeatmapData {
  width: number;
  height: number;
  points: HeatmapPoint[];
  max_value: number;
}

export interface AnalysisResponse {
  id: string;
  filename: string;
  status: string;
  progress: number;
  error_message?: string | null;
  duration_seconds: number | null;
  resolution: string | null;
  fps: number | null;
  total_frames_analyzed: number | null;
  overall_score: number | null;
  crosshair_score: number | null;
  movement_score: number | null;
  decision_score: number | null;
  communication_score: number | null;
  map_score: number | null;
  status_text?: string | null;
  crosshair_data: CrosshairData | null;
  movement_data: MovementData | null;
  decision_data: DecisionData | null;
  communication_data: CommunicationData | null;
  map_data: MapData | null;
  timeline_events: TimelineEvent[] | null;
  recommendations: Recommendation[] | null;
  heatmap_data: HeatmapData | null;
  round_analysis: RoundAnalysis[] | null;
  pro_comparison: ProComparison | null;
  created_at: string | null;
  updated_at: string | null;
}
