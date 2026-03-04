import type { AnalysisResponse, AnalysisListItem } from "../types/analysis";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function uploadVod(
  file: File,
  onUploadProgress?: (pct: number) => void
): Promise<{ id: string; message: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

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
          reject(new Error("Invalid response"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || "Upload failed"));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));

    xhr.open("POST", `${API_URL}/api/upload`);
    xhr.send(formData);
  });
}

export async function listAnalyses(): Promise<AnalysisListItem[]> {
  const res = await fetch(`${API_URL}/api/analyses`);
  if (!res.ok) throw new Error("Failed to fetch analyses");
  return res.json();
}

export async function getAnalysis(id: string): Promise<AnalysisResponse> {
  const res = await fetch(`${API_URL}/api/analysis/${id}`);
  if (!res.ok) throw new Error("Failed to fetch analysis");
  return res.json();
}

export interface AnalysisStatusResponse {
  id: string;
  status: string;
  progress: number;
  status_text: string | null;
  error_message: string | null;
}

export async function getAnalysisStatus(id: string): Promise<AnalysisStatusResponse> {
  const res = await fetch(`${API_URL}/api/analysis/${id}/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function getDemoAnalysis(): Promise<AnalysisResponse> {
  const res = await fetch(`${API_URL}/api/demo-analysis`);
  if (!res.ok) throw new Error("Failed to fetch demo");
  return res.json();
}

export async function deleteAnalysis(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/analysis/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete analysis");
}
