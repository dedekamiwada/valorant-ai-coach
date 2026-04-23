import { useState, useEffect, useCallback, useRef } from "react";
import {
  Upload,
  Trash2,
  Database,
  User,
  Trophy,
  Loader2,
  AlertTriangle,
  X,
  Film,
  HardDrive,
  Clock,
  Filter,
  Brain,
  Sparkles,
} from "lucide-react";
import type { DatasetListItem } from "../types/dataset";
import {
  uploadDataset,
  listDatasets,
  deleteDataset,
  getDatasetStats,
  startDatasetAnalysis,
} from "../api/datasets";
import type { DatasetStats } from "../types/dataset";

const AGENTS = [
  "Jett", "Raze", "Phoenix", "Reyna", "Yoru", "Neon", "Iso", "Waylay",
  "Sage", "Cypher", "Killjoy", "Chamber", "Deadlock",
  "Sova", "Breach", "Skye", "KAY/O", "Fade", "Gekko",
  "Brimstone", "Viper", "Omen", "Astra", "Harbor", "Clove",
];

const MAPS = [
  "Bind", "Haven", "Ascent", "Split", "Icebox",
  "Breeze", "Fracture", "Pearl", "Lotus", "Sunset", "Corrode",
];

function formatBytes(bytes: number | null): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DatasetManager() {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Upload form fields
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formSource, setFormSource] = useState<"user" | "pro">("user");
  const [formPlayer, setFormPlayer] = useState("");
  const [formTeam, setFormTeam] = useState("");
  const [formAgent, setFormAgent] = useState("");
  const [formMap, setFormMap] = useState("");
  const [formTags, setFormTags] = useState("");

  // Filter state
  const [filterSource, setFilterSource] = useState("");
  const [filterAgent, setFilterAgent] = useState("");
  const [filterMap, setFilterMap] = useState("");

  // Delete confirmation
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Per-dataset "analyze" action state (id → inflight request flag)
  const [analyzingIds, setAnalyzingIds] = useState<Record<string, boolean>>({});

  const refresh = useCallback(async () => {
    try {
      const params: { source?: string; agent?: string; map_name?: string } = {};
      if (filterSource) params.source = filterSource;
      if (filterAgent) params.agent = filterAgent;
      if (filterMap) params.map_name = filterMap;

      const [datasetList, datasetStats] = await Promise.all([
        listDatasets(params),
        getDatasetStats(),
      ]);
      setDatasets(datasetList);
      setStats(datasetStats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar datasets");
    } finally {
      setLoading(false);
    }
  }, [filterSource, filterAgent, filterMap]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // While any dataset is in "analyzing" state, poll the list every 3s so the
  // user sees progress updates without having to refresh manually.
  useEffect(() => {
    const anyAnalyzing = datasets.some((d) => d.status === "analyzing");
    if (!anyAnalyzing) return;
    const interval = window.setInterval(() => {
      refresh();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [datasets, refresh]);

  const handleAnalyze = useCallback(
    async (id: string) => {
      setAnalyzingIds((prev) => ({ ...prev, [id]: true }));
      setError(null);
      try {
        await startDatasetAnalysis(id);
        await refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Falha ao iniciar análise");
      } finally {
        setAnalyzingIds((prev) => {
          const next = { ...prev };
          delete next[id];
          return next;
        });
      }
    },
    [refresh]
  );

  const handleFileSelect = useCallback((file: File) => {
    setUploadFile(file);
    if (!formName) {
      setFormName(file.name.replace(/\.[^.]+$/, ""));
    }
    setShowUploadForm(true);
  }, [formName]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const handleUpload = async () => {
    if (!uploadFile || !formName) return;

    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      await uploadDataset(
        {
          file: uploadFile,
          name: formName,
          description: formDescription || undefined,
          source: formSource,
          player_name: formPlayer || undefined,
          team: formTeam || undefined,
          agent: formAgent || undefined,
          map_name: formMap || undefined,
          tags: formTags || undefined,
        },
        (pct) => setUploadProgress(pct)
      );

      // Reset form
      setUploading(false);
      setUploadProgress(0);
      setShowUploadForm(false);
      setUploadFile(null);
      setFormName("");
      setFormDescription("");
      setFormSource("user");
      setFormPlayer("");
      setFormTeam("");
      setFormAgent("");
      setFormMap("");
      setFormTags("");

      // Refresh list
      await refresh();
    } catch (err) {
      setUploading(false);
      setUploadProgress(0);
      setError(err instanceof Error ? err.message : "Upload falhou");
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteDataset(id);
      setDeletingId(null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao deletar");
    }
  };

  return (
    <div className="min-h-screen p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center">
            <Database className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Datasets</h1>
            <p className="text-sm text-gray-500">VODs de referência para treinamento da AI</p>
          </div>
        </div>
        <button
          onClick={() => {
            setShowUploadForm(true);
            setUploadFile(null);
          }}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl gradient-valorant text-white font-medium hover:opacity-90 transition-opacity"
        >
          <Upload className="w-4 h-4" />
          Upload Dataset
        </button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <Film className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Total</span>
            </div>
            <span className="text-2xl font-bold text-white">{stats.total_datasets}</span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <Trophy className="w-4 h-4 text-yellow-500" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Pro VODs</span>
            </div>
            <span className="text-2xl font-bold text-yellow-400">{stats.pro_datasets}</span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <User className="w-4 h-4 text-blue-500" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Seus VODs</span>
            </div>
            <span className="text-2xl font-bold text-blue-400">{stats.user_datasets}</span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <HardDrive className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Armazenamento</span>
            </div>
            <span className="text-2xl font-bold text-white">{formatBytes(stats.total_size_bytes)}</span>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="flex items-center gap-2 text-gray-500">
          <Filter className="w-4 h-4" />
          <span className="text-sm">Filtros:</span>
        </div>
        <select
          value={filterSource}
          onChange={(e) => setFilterSource(e.target.value)}
          className="bg-gray-800/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-white/20"
        >
          <option value="">Todas as fontes</option>
          <option value="pro">Pro Players</option>
          <option value="user">Meus VODs</option>
        </select>
        <select
          value={filterAgent}
          onChange={(e) => setFilterAgent(e.target.value)}
          className="bg-gray-800/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-white/20"
        >
          <option value="">Todos os agentes</option>
          {AGENTS.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
        <select
          value={filterMap}
          onChange={(e) => setFilterMap(e.target.value)}
          className="bg-gray-800/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-white/20"
        >
          <option value="">Todos os mapas</option>
          {MAPS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
          <button onClick={() => setError(null)} className="ml-auto">
            <X className="w-4 h-4 text-red-400" />
          </button>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Upload Dataset</h2>
              <button
                onClick={() => {
                  setShowUploadForm(false);
                  setUploadFile(null);
                }}
                disabled={uploading}
              >
                <X className="w-5 h-5 text-gray-400 hover:text-white" />
              </button>
            </div>

            {/* File drop zone */}
            {!uploadFile && (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  border-2 border-dashed rounded-xl p-8 text-center cursor-pointer mb-4
                  transition-all duration-300
                  ${isDragging ? "border-blue-500 bg-blue-500/5" : "border-white/10 hover:border-white/20"}
                `}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp4,.avi,.mkv,.mov,.webm"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileSelect(file);
                  }}
                />
                <Upload className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                <p className="text-gray-300 mb-1">Arraste o VOD aqui ou clique</p>
                <p className="text-xs text-gray-600">MP4, AVI, MKV, MOV, WebM</p>
              </div>
            )}

            {/* Selected file info */}
            {uploadFile && (
              <div className="bg-gray-800/50 rounded-lg p-3 mb-4 flex items-center gap-3">
                <Film className="w-5 h-5 text-blue-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{uploadFile.name}</p>
                  <p className="text-xs text-gray-500">{formatBytes(uploadFile.size)}</p>
                </div>
                <button
                  onClick={() => setUploadFile(null)}
                  disabled={uploading}
                  className="text-gray-500 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Form fields */}
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Nome *</label>
                <input
                  type="text"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="Ex: TenZ Jett Ascent"
                  className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                />
              </div>

              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Descrição</label>
                <input
                  type="text"
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  placeholder="Ex: Ranked Immortal gameplay"
                  className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Fonte</label>
                  <select
                    value={formSource}
                    onChange={(e) => setFormSource(e.target.value as "user" | "pro")}
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/20"
                  >
                    <option value="user">Meu VOD</option>
                    <option value="pro">Pro Player</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Agente</label>
                  <select
                    value={formAgent}
                    onChange={(e) => setFormAgent(e.target.value)}
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/20"
                  >
                    <option value="">Selecionar...</option>
                    {AGENTS.map((a) => (
                      <option key={a} value={a}>{a}</option>
                    ))}
                  </select>
                </div>
              </div>

              {formSource === "pro" && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Jogador</label>
                    <input
                      type="text"
                      value={formPlayer}
                      onChange={(e) => setFormPlayer(e.target.value)}
                      placeholder="Ex: TenZ, Aspas, nAts"
                      className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Time</label>
                    <input
                      type="text"
                      value={formTeam}
                      onChange={(e) => setFormTeam(e.target.value)}
                      placeholder="Ex: SEN, LOUD, FNC"
                      className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                    />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Mapa</label>
                  <select
                    value={formMap}
                    onChange={(e) => setFormMap(e.target.value)}
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/20"
                  >
                    <option value="">Selecionar...</option>
                    {MAPS.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Tags</label>
                  <input
                    type="text"
                    value={formTags}
                    onChange={(e) => setFormTags(e.target.value)}
                    placeholder="ranked, clutch, pistol"
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                  />
                </div>
              </div>
            </div>

            {/* Upload progress */}
            {uploading && (
              <div className="mt-4">
                <div className="flex items-center gap-3 mb-2">
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  <span className="text-sm text-gray-300">Enviando... {uploadProgress}%</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowUploadForm(false);
                  setUploadFile(null);
                }}
                disabled={uploading}
                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white border border-white/10 hover:border-white/20 transition-all"
              >
                Cancelar
              </button>
              <button
                onClick={handleUpload}
                disabled={!uploadFile || !formName || uploading}
                className="px-5 py-2 rounded-lg text-sm font-medium text-white gradient-valorant hover:opacity-90 disabled:opacity-40 transition-all"
              >
                {uploading ? "Enviando..." : "Upload"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {deletingId && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-sm p-6">
            <h3 className="text-lg font-semibold text-white mb-2">Deletar Dataset?</h3>
            <p className="text-sm text-gray-400 mb-6">
              Esta ação não pode ser desfeita. O arquivo será removido permanentemente.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeletingId(null)}
                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white border border-white/10"
              >
                Cancelar
              </button>
              <button
                onClick={() => handleDelete(deletingId)}
                className="px-4 py-2 rounded-lg text-sm font-medium text-white bg-red-600 hover:bg-red-700 transition-colors"
              >
                Deletar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dataset List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
        </div>
      ) : datasets.length === 0 ? (
        <div className="text-center py-20">
          <Database className="w-16 h-16 text-gray-700 mx-auto mb-4" />
          <p className="text-lg text-gray-400 mb-2">Nenhum dataset ainda</p>
          <p className="text-sm text-gray-600">
            Faça upload de VODs de referência para treinar a AI
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {datasets.map((d) => (
            <div
              key={d.id}
              className="gradient-card rounded-xl border border-white/5 hover:border-white/10 transition-colors p-4"
            >
              <div className="flex items-center gap-4">
                {/* Icon */}
                <div
                  className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                    d.source === "pro"
                      ? "bg-yellow-500/10"
                      : "bg-blue-500/10"
                  }`}
                >
                  {d.source === "pro" ? (
                    <Trophy className="w-5 h-5 text-yellow-400" />
                  ) : (
                    <User className="w-5 h-5 text-blue-400" />
                  )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-white truncate">{d.name}</h3>
                    {d.player_name && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-500/10 text-yellow-400">
                        {d.player_name}
                      </span>
                    )}
                    {d.agent && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400">
                        {d.agent}
                      </span>
                    )}
                    {d.map_name && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/10 text-green-400">
                        {d.map_name}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                    <span>{d.filename}</span>
                    <span>{formatBytes(d.file_size_bytes)}</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatDate(d.created_at)}
                    </span>
                  </div>
                </div>

                {/* Analysis status / action */}
                <div className="flex flex-col items-end gap-1.5 min-w-[10rem]">
                  <span
                    className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                      d.status === "ready"
                        ? "bg-emerald-500/10 text-emerald-400"
                        : d.status === "uploaded"
                        ? "bg-green-500/10 text-green-400"
                        : d.status === "analyzing"
                        ? "bg-yellow-500/10 text-yellow-400"
                        : d.status === "failed"
                        ? "bg-red-500/10 text-red-400"
                        : "bg-blue-500/10 text-blue-400"
                    }`}
                  >
                    {d.status === "ready"
                      ? "pontos fortes prontos"
                      : d.status === "analyzing"
                      ? `analisando ${d.analysis_progress ?? 0}%`
                      : d.status}
                  </span>
                  {d.status === "analyzing" && (
                    <div className="w-40 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-yellow-400 rounded-full transition-all"
                        style={{ width: `${d.analysis_progress ?? 0}%` }}
                      />
                    </div>
                  )}
                  {d.analysis_status_text && d.status === "analyzing" && (
                    <span className="text-[11px] text-gray-500 text-right max-w-[12rem] truncate">
                      {d.analysis_status_text}
                    </span>
                  )}
                </div>

                {/* Analyze (pro VOD) */}
                {d.source === "pro" &&
                  (d.status === "uploaded" || d.status === "failed") && (
                    <button
                      onClick={() => handleAnalyze(d.id)}
                      disabled={!!analyzingIds[d.id]}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-yellow-300 bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/20 disabled:opacity-40 transition-colors"
                      title="Rodar análise de IA neste VOD de pro e extrair pontos fortes"
                    >
                      {analyzingIds[d.id] ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Sparkles className="w-3.5 h-3.5" />
                      )}
                      Analisar VOD
                    </button>
                  )}

                {d.status === "ready" && (
                  <span
                    className="inline-flex items-center gap-1 text-xs text-emerald-300"
                    title="Entradas geradas na Knowledge Base a partir desta análise"
                  >
                    <Brain className="w-3.5 h-3.5" />
                    Ver pontos fortes
                  </span>
                )}

                {/* Delete */}
                <button
                  onClick={() => setDeletingId(d.id)}
                  className="text-gray-600 hover:text-red-400 transition-colors p-2"
                  title="Deletar"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
