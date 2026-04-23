export interface DatasetListItem {
  id: string;
  name: string;
  source: string;
  player_name: string | null;
  agent: string | null;
  map_name: string | null;
  filename: string;
  file_size_bytes: number | null;
  duration_seconds: number | null;
  analysis_id: string | null;
  status: string;
  analysis_progress: number | null;
  analysis_status_text: string | null;
  created_at: string | null;
}

export interface DatasetResponse {
  id: string;
  name: string;
  description: string | null;
  source: string;
  player_name: string | null;
  team: string | null;
  agent: string | null;
  map_name: string | null;
  rank: string | null;
  tags: string[] | null;
  filename: string;
  file_size_bytes: number | null;
  duration_seconds: number | null;
  analysis_id: string | null;
  status: string;
  analysis_progress: number | null;
  analysis_status_text: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface DatasetAnalysisStatus {
  id: string;
  status: string;
  progress: number;
  status_text: string | null;
  analysis_id: string | null;
  knowledge_entries_count: number | null;
}

export interface DatasetStats {
  total_datasets: number;
  pro_datasets: number;
  user_datasets: number;
  total_size_bytes: number;
  total_duration_seconds: number;
  agents: Record<string, number>;
  maps: Record<string, number>;
}

export interface DatasetUploadPayload {
  file: File;
  name: string;
  description?: string;
  source?: string;
  player_name?: string;
  team?: string;
  agent?: string;
  map_name?: string;
  rank?: string;
  tags?: string;
}
