import type {
  DatasetAnalysisStatus,
  DatasetListItem,
  DatasetResponse,
  DatasetStats,
  DatasetUploadPayload,
} from "../types/dataset";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function uploadDataset(
  payload: DatasetUploadPayload,
  onUploadProgress?: (pct: number) => void
): Promise<{ id: string; message: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", payload.file);
    formData.append("name", payload.name);
    if (payload.description) formData.append("description", payload.description);
    if (payload.source) formData.append("source", payload.source);
    if (payload.player_name) formData.append("player_name", payload.player_name);
    if (payload.team) formData.append("team", payload.team);
    if (payload.agent) formData.append("agent", payload.agent);
    if (payload.map_name) formData.append("map_name", payload.map_name);
    if (payload.rank) formData.append("rank", payload.rank);
    if (payload.tags) formData.append("tags", payload.tags);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onUploadProgress) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onUploadProgress(pct);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Resposta inválida"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || "Upload falhou"));
        } catch {
          reject(new Error(`Upload falhou (${xhr.status})`));
        }
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Erro de rede")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelado")));

    xhr.open("POST", `${API_URL}/api/datasets/upload`);
    xhr.send(formData);
  });
}

export async function listDatasets(params?: {
  source?: string;
  agent?: string;
  map_name?: string;
}): Promise<DatasetListItem[]> {
  const searchParams = new URLSearchParams();
  if (params?.source) searchParams.set("source", params.source);
  if (params?.agent) searchParams.set("agent", params.agent);
  if (params?.map_name) searchParams.set("map_name", params.map_name);

  const qs = searchParams.toString();
  const url = `${API_URL}/api/datasets${qs ? `?${qs}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Falha ao listar datasets");
  return res.json();
}

export async function getDataset(id: string): Promise<DatasetResponse> {
  const res = await fetch(`${API_URL}/api/datasets/${id}`);
  if (!res.ok) throw new Error("Falha ao buscar dataset");
  return res.json();
}

export async function deleteDataset(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/datasets/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Falha ao deletar dataset");
}

export async function getDatasetStats(): Promise<DatasetStats> {
  const res = await fetch(`${API_URL}/api/datasets/stats/summary`);
  if (!res.ok) throw new Error("Falha ao buscar estatísticas");
  return res.json();
}

export async function startDatasetAnalysis(
  id: string
): Promise<{ id: string; message: string }> {
  const res = await fetch(`${API_URL}/api/datasets/${id}/analyze`, {
    method: "POST",
  });
  if (!res.ok) {
    try {
      const err = await res.json();
      throw new Error(err.detail || "Falha ao iniciar análise");
    } catch {
      throw new Error(`Falha ao iniciar análise (${res.status})`);
    }
  }
  return res.json();
}

export async function getDatasetAnalysisStatus(
  id: string
): Promise<DatasetAnalysisStatus> {
  const res = await fetch(`${API_URL}/api/datasets/${id}/analysis-status`);
  if (!res.ok) throw new Error("Falha ao buscar status da análise");
  return res.json();
}
