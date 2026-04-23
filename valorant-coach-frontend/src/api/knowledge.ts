import type { KnowledgeListItem, KnowledgeResponse, KnowledgeStats, KnowledgeCreate } from "../types/knowledge";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function listKnowledge(params?: {
  category?: string;
  agent?: string;
  map_name?: string;
  source_type?: string;
}): Promise<KnowledgeListItem[]> {
  const searchParams = new URLSearchParams();
  if (params?.category) searchParams.set("category", params.category);
  if (params?.agent) searchParams.set("agent", params.agent);
  if (params?.map_name) searchParams.set("map_name", params.map_name);
  if (params?.source_type) searchParams.set("source_type", params.source_type);

  const qs = searchParams.toString();
  const url = `${API_URL}/api/knowledge${qs ? `?${qs}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Falha ao listar knowledge base");
  return res.json();
}

export async function getKnowledgeEntry(id: string): Promise<KnowledgeResponse> {
  const res = await fetch(`${API_URL}/api/knowledge/${id}`);
  if (!res.ok) throw new Error("Falha ao buscar entry");
  return res.json();
}

export async function createKnowledgeEntry(entry: KnowledgeCreate): Promise<KnowledgeResponse> {
  const res = await fetch(`${API_URL}/api/knowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  if (!res.ok) throw new Error("Falha ao criar entry");
  return res.json();
}

export async function deleteKnowledgeEntry(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/knowledge/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Falha ao deletar entry");
}

export async function getKnowledgeStats(): Promise<KnowledgeStats> {
  const res = await fetch(`${API_URL}/api/knowledge/stats`);
  if (!res.ok) throw new Error("Falha ao buscar estatísticas");
  return res.json();
}

export async function extractKnowledgeFromAnalysis(analysisId: string): Promise<{
  analysis_id: string;
  entries_created: number;
  entries: string[];
}> {
  const res = await fetch(`${API_URL}/api/knowledge/extract-from-analysis/${analysisId}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Falha ao extrair knowledge");
  return res.json();
}
