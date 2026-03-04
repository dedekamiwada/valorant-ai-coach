export interface KnowledgeListItem {
  id: string;
  source_type: string;
  category: string;
  subcategory: string | null;
  agent: string | null;
  map_name: string | null;
  title: string;
  confidence: number;
  observation_count: number;
  created_at: string | null;
}

export interface KnowledgeResponse {
  id: string;
  source_type: string;
  source_id: string | null;
  category: string;
  subcategory: string | null;
  agent: string | null;
  map_name: string | null;
  rank: string | null;
  title: string;
  description: string;
  metric_name: string | null;
  metric_value: number | null;
  confidence: number;
  observation_count: number;
  tags: string[] | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface KnowledgeStats {
  total_entries: number;
  avg_confidence: number;
  categories: Record<string, number>;
  agents: Record<string, number>;
  maps: Record<string, number>;
  sources: Record<string, number>;
}

export interface KnowledgeCreate {
  source_type?: string;
  category: string;
  subcategory?: string;
  agent?: string;
  map_name?: string;
  rank?: string;
  title: string;
  description: string;
  metric_name?: string;
  metric_value?: number;
  confidence?: number;
  tags?: string[];
}
