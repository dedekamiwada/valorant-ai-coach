import { useState, useEffect, useCallback, useRef } from "react";
import {
  Upload,
  Target,
  Activity,
  Brain,
  MessageSquare,
  TrendingUp,
  AlertTriangle,
  ChevronRight,
  Clock,
  Crosshair,
  Zap,
  Shield,
  Eye,
  BarChart3,
  Loader2,
  Trophy,
  Star,
  X,
  MapPin,
  Navigation,
  Timer,
} from "lucide-react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import type { AnalysisResponse } from "./types/analysis";
import { getDemoAnalysis, uploadVod, getAnalysisStatus, getAnalysis } from "./api/client";
import type { AnalysisStatusResponse } from "./api/client";
import DatasetManager from "./components/DatasetManager";
import KnowledgeBase from "./components/KnowledgeBase";

type AppPage = "analysis" | "datasets" | "knowledge";

// ─── Score Ring Component ───
function ScoreRing({
  score,
  size = 120,
  strokeWidth = 8,
  label,
  color,
}: {
  score: number;
  size?: number;
  strokeWidth?: number;
  label: string;
  color?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const scoreColor =
    color || (score >= 80 ? "#22c55e" : score >= 60 ? "#eab308" : score >= 40 ? "#f97316" : "#ef4444");

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={scoreColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold" style={{ color: scoreColor }}>
            {score.toFixed(0)}
          </span>
        </div>
      </div>
      <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</span>
    </div>
  );
}

// ─── Stat Card ───
function StatCard({
  icon: Icon,
  label,
  value,
  unit,
  status,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  unit?: string;
  status?: "good" | "warn" | "bad";
}) {
  const statusColors = {
    good: "text-green-400",
    warn: "text-yellow-400",
    bad: "text-red-400",
  };

  return (
    <div className="gradient-card rounded-xl border border-white/5 p-4 hover:border-white/10 transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-gray-500" />
        <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold ${status ? statusColors[status] : "text-white"}`}>
          {value}
        </span>
        {unit && <span className="text-sm text-gray-500">{unit}</span>}
      </div>
    </div>
  );
}

// ─── Section Header ───
function SectionHeader({
  icon: Icon,
  title,
  subtitle,
  score,
  weight,
}: {
  icon: React.ElementType;
  title: string;
  subtitle: string;
  score?: number;
  weight?: string;
}) {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
          <Icon className="w-5 h-5 text-red-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <p className="text-xs text-gray-500">{subtitle}</p>
        </div>
      </div>
      {score !== undefined && (
        <div className="text-right">
          <span
            className={`text-2xl font-bold ${
              score >= 80 ? "text-green-400" : score >= 60 ? "text-yellow-400" : "text-red-400"
            }`}
          >
            {score.toFixed(1)}
          </span>
          <span className="text-sm text-gray-500">/100</span>
          {weight && <p className="text-xs text-gray-600">{weight}</p>}
        </div>
      )}
    </div>
  );
}

// ─── Heatmap Canvas Component ───
function CrosshairHeatmap({ data }: { data: AnalysisResponse["heatmap_data"] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !data) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const scaleX = w / data.width;
    const scaleY = h / data.height;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, w, h);

    // Draw grid
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw head level zone
    const headTop = h * 0.25;
    const headBottom = h * 0.45;
    ctx.fillStyle = "rgba(34, 197, 94, 0.05)";
    ctx.fillRect(0, headTop, w, headBottom - headTop);
    ctx.strokeStyle = "rgba(34, 197, 94, 0.2)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, headTop);
    ctx.lineTo(w, headTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, headBottom);
    ctx.lineTo(w, headBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "rgba(34, 197, 94, 0.5)";
    ctx.fillText("HEAD LEVEL ZONE", 8, headTop - 4);

    // Draw heatmap points
    const maxVal = data.max_value || 1;
    for (const point of data.points) {
      const px = point.x * scaleX;
      const py = point.y * scaleY;
      const intensity = point.value / maxVal;

      const r = 4 + intensity * 12;
      const gradient = ctx.createRadialGradient(px, py, 0, px, py, r);

      if (py >= headTop && py <= headBottom) {
        gradient.addColorStop(0, `rgba(34, 197, 94, ${0.4 * intensity})`);
        gradient.addColorStop(1, "rgba(34, 197, 94, 0)");
      } else if (py > h * 0.6) {
        gradient.addColorStop(0, `rgba(239, 68, 68, ${0.4 * intensity})`);
        gradient.addColorStop(1, "rgba(239, 68, 68, 0)");
      } else {
        gradient.addColorStop(0, `rgba(234, 179, 8, ${0.4 * intensity})`);
        gradient.addColorStop(1, "rgba(234, 179, 8, 0)");
      }

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Crosshair center
    const cx = w / 2;
    const cy = h / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();
  }, [data]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={480}
        height={270}
        className="w-full rounded-lg border border-white/5"
      />
      <div className="absolute bottom-2 right-2 flex gap-2 text-xs">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-500" /> Head Level
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-yellow-500" /> Off-level
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500" /> Floor
        </span>
      </div>
    </div>
  );
}

// ─── Upload Component ───
function UploadSection({
  onAnalysisReady,
  onLoadDemo,
}: {
  onAnalysisReady: (data: AnalysisResponse) => void;
  onLoadDemo: () => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startPolling = useCallback(
    (analysisId: string) => {
      setProcessing(true);
      const pollInterval = setInterval(async () => {
        try {
          const status: AnalysisStatusResponse = await getAnalysisStatus(analysisId);
          setProgress(status.progress);

          if (status.status === "processing" && status.status_text) {
            setStatusText(status.status_text);
          } else if (status.status === "processing") {
            setStatusText("Analisando gameplay...");
          }

          if (status.status === "completed") {
            clearInterval(pollInterval);
            setStatusText("Carregando resultados...");
            const analysisResult = await getAnalysis(analysisId);
            setProcessing(false);
            setStatusText("");
            onAnalysisReady(analysisResult);
          } else if (status.status === "failed") {
            clearInterval(pollInterval);
            setProcessing(false);
            setStatusText("");
            setError(status.error_message || "Análise falhou");
          }
        } catch {
          clearInterval(pollInterval);
          setProcessing(false);
          setStatusText("");
          setError("Falha ao verificar status");
        }
      }, 1500);
    },
    [onAnalysisReady]
  );

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setUploading(true);
      setUploadProgress(0);

      try {
        const result = await uploadVod(file, (pct) => {
          setUploadProgress(pct);
        });
        setUploading(false);
        setUploadProgress(100);
        setStatusText("Iniciando análise...");
        startPolling(result.id);
      } catch (err) {
        setUploading(false);
        setUploadProgress(0);
        setError(err instanceof Error ? err.message : "Upload falhou");
      }
    },
    [startPolling]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const isWorking = uploading || processing;

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Logo/Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-14 h-14 rounded-2xl gradient-valorant flex items-center justify-center glow-red">
              <Crosshair className="w-7 h-7 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            Valorant <span className="text-red-500">AI Coach</span>
          </h1>
          <p className="text-gray-400 text-lg">
            Upload your VOD for professional-grade gameplay analysis
          </p>
        </div>

        {/* File Upload Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => !isWorking && fileInputRef.current?.click()}
            className={`
              relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
              transition-all duration-300
              ${isDragging ? "border-red-500 bg-red-500/5" : "border-white/10 hover:border-white/20"}
              ${isWorking ? "pointer-events-none opacity-60" : ""}
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mkv,.mov,.webm"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />

            {uploading ? (
              <div className="flex flex-col items-center gap-4">
                <div className="relative w-16 h-16">
                  <Loader2 className="w-16 h-16 text-red-400 animate-spin" />
                  <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-red-400">
                    {uploadProgress}%
                  </span>
                </div>
                <p className="text-lg text-gray-300">Enviando vídeo...</p>
                <div className="w-64 h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-sm text-gray-500">
                  {uploadProgress < 100 ? `${uploadProgress}% enviado` : "Upload completo, iniciando análise..."}
                </p>
              </div>
            ) : processing ? (
              <div className="flex flex-col items-center gap-4">
                <div className="relative w-16 h-16">
                  <Loader2 className="w-16 h-16 text-red-400 animate-spin" />
                  <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-red-400">
                    {progress}%
                  </span>
                </div>
                <p className="text-lg text-gray-300 font-medium">{statusText || "Processando..."}</p>
                <div className="w-80 h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full gradient-valorant rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-sm text-gray-500">
                  {progress}% concluído
                </p>
              </div>
            ) : (
              <>
                <Upload className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-lg text-gray-300 mb-2">
                  Drag & drop your VOD here or click to browse
                </p>
                <p className="text-sm text-gray-600">
                  Supports MP4, AVI, MKV, MOV, WebM
                </p>
              </>
            )}
          </div>

        {error && (
          <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
            <p className="text-red-300 text-sm">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto">
              <X className="w-4 h-4 text-red-400" />
            </button>
          </div>
        )}

        {/* Demo Button */}
        <div className="mt-6 text-center">
          <button
            onClick={onLoadDemo}
            disabled={isWorking}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all text-gray-300 hover:text-white disabled:opacity-40"
          >
            <Eye className="w-4 h-4" />
            View Demo Analysis
          </button>
          <p className="text-xs text-gray-600 mt-2">See a sample analysis with realistic data</p>
        </div>

        {/* Features */}
        <div className="mt-12 grid grid-cols-2 gap-4">
          {[
            { icon: Crosshair, title: "Crosshair Placement", desc: "Head level tracking & pre-aim analysis" },
            { icon: Activity, title: "Movement Analysis", desc: "Counter-strafe & peek detection" },
            { icon: Brain, title: "Decision Making", desc: "Exposure & trade efficiency" },
            { icon: MessageSquare, title: "Comunicação", desc: "Timing e precisão das callouts" },
          ].map((f) => (
            <div key={f.title} className="gradient-card rounded-xl border border-white/5 p-4">
              <f.icon className="w-5 h-5 text-red-400 mb-2" />
              <h3 className="text-sm font-medium text-white mb-1">{f.title}</h3>
              <p className="text-xs text-gray-500">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Dashboard Component ───
function Dashboard({
  analysis,
  onBack,
}: {
  analysis: AnalysisResponse;
  onBack: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"overview" | "crosshair" | "movement" | "decision" | "map" | "comms" | "rounds">(
    "overview"
  );

  const crosshair = analysis.crosshair_data;
  const movement = analysis.movement_data;
  const decision = analysis.decision_data;
  const comms = analysis.communication_data;
  const mapData = analysis.map_data;
  const proComp = analysis.pro_comparison;

  // Radar chart data
  const radarData = proComp
    ? [
        { metric: "Crosshair", player: proComp.player_crosshair, nAts: proComp.benchmarks.nAts.crosshair, TenZ: proComp.benchmarks.TenZ.crosshair },
        { metric: "Movement", player: proComp.player_movement, nAts: proComp.benchmarks.nAts.movement, TenZ: proComp.benchmarks.TenZ.movement },
        { metric: "Decision", player: proComp.player_decision, nAts: proComp.benchmarks.nAts.decision, TenZ: proComp.benchmarks.TenZ.decision },
        { metric: "Map", player: proComp.player_map || 0, nAts: proComp.benchmarks.nAts.map || 90, TenZ: proComp.benchmarks.TenZ.map || 80 },
        { metric: "Comms", player: proComp.player_communication, nAts: proComp.benchmarks.nAts.communication, TenZ: proComp.benchmarks.TenZ.communication },
      ]
    : [];

  // Zone distribution for map tab
  const zoneColors: Record<string, string> = {
    a_site: "#22c55e",
    b_site: "#3b82f6",
    mid: "#eab308",
    spawn: "#ef4444",
    unknown: "#6b7280",
  };
  const zoneLabels: Record<string, string> = {
    a_site: "A Site",
    b_site: "B Site",
    mid: "Mid",
    spawn: "Spawn",
    unknown: "Desconhecido",
  };
  const zonePieData = mapData
    ? Object.entries(mapData.time_in_zones)
        .filter(([, v]) => v > 0)
        .map(([zone, value]) => ({
          name: zoneLabels[zone] || zone,
          value: Math.round(value * 10) / 10,
          fill: zoneColors[zone] || "#6b7280",
        }))
    : [];

  // Peek distribution pie data
  const peekData = movement
    ? Object.entries(movement.peek_type_distribution).map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
      }))
    : [];
  const PEEK_COLORS = ["#22c55e", "#eab308", "#ef4444"];

  // Round analysis bar chart
  const roundChartData =
    analysis.round_analysis?.map((r) => ({
      round: `R${r.round_number}`,
      crosshair: r.crosshair_score,
      movement: r.movement_score,
      decision: r.decision_score,
      outcome: r.outcome,
    })) || [];

  // Timeline data for area chart
  const crosshairTimeData =
    crosshair?.frame_data?.slice(0, 80).map((f) => ({
      time: f.timestamp.toFixed(0),
      headLevel: f.head_level ? 100 : 0,
      adjustment: f.adjustment,
    })) || [];

  const tabs = [
    { id: "overview" as const, label: "Overview", icon: BarChart3 },
    { id: "crosshair" as const, label: "Crosshair", icon: Crosshair },
    { id: "movement" as const, label: "Movement", icon: Activity },
    { id: "decision" as const, label: "Decision", icon: Brain },
    { id: "map" as const, label: "Mapa", icon: MapPin },
    { id: "comms" as const, label: "Comms", icon: MessageSquare },
    { id: "rounds" as const, label: "Rounds", icon: Clock },
  ];

  return (
    <div className="min-h-screen">
      {/* Top Bar */}
      <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <div className="w-8 h-8 rounded-lg gradient-valorant flex items-center justify-center">
                <Crosshair className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-white">AI Coach</span>
            </button>
            <ChevronRight className="w-4 h-4 text-gray-600" />
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-300">{analysis.filename}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {analysis.duration_seconds && (
              <span className="text-xs text-gray-500">
                {Math.floor(analysis.duration_seconds / 60)}m {Math.floor(analysis.duration_seconds % 60)}s
              </span>
            )}
            {analysis.resolution && <span className="text-xs text-gray-500">{analysis.resolution}</span>}
          </div>
        </div>
      </header>

      {/* Score Banner */}
      <div className="bg-gradient-to-r from-gray-900 via-gray-950 to-gray-900 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center gap-12">
            <ScoreRing score={analysis.overall_score || 0} size={140} strokeWidth={10} label="Overall" />
            <div className="flex-1 grid grid-cols-5 gap-4">
              <ScoreRing score={analysis.crosshair_score || 0} size={80} label="Crosshair (55%)" />
              <ScoreRing score={analysis.movement_score || 0} size={80} label="Movement (18%)" />
              <ScoreRing score={analysis.decision_score || 0} size={80} label="Decision (12%)" />
              <ScoreRing score={analysis.map_score || 0} size={80} label="Mapa (10%)" />
              <ScoreRing score={analysis.communication_score || 0} size={80} label="Comms (5%)" />
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-white/5 bg-gray-950/50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2
                  ${activeTab === tab.id ? "border-red-500 text-white" : "border-transparent text-gray-500 hover:text-gray-300"}
                `}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* ─── OVERVIEW TAB ─── */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            {/* Recommendations */}
            {analysis.recommendations && analysis.recommendations.length > 0 && (
              <div>
                <SectionHeader
                  icon={AlertTriangle}
                  title="Priority Recommendations"
                  subtitle="Top areas for improvement based on your gameplay"
                />
                <div className="space-y-4">
                  {analysis.recommendations.map((rec, i) => (
                    <div
                      key={i}
                      className={`gradient-card rounded-xl border p-5 ${
                        rec.priority === 1 ? "border-red-500/30" : rec.priority === 2 ? "border-yellow-500/20" : "border-white/5"
                      }`}
                    >
                      <div className="flex items-start gap-4">
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                            rec.priority === 1 ? "bg-red-500/20" : rec.priority === 2 ? "bg-yellow-500/20" : "bg-blue-500/20"
                          }`}
                        >
                          <span
                            className={`text-sm font-bold ${
                              rec.priority === 1 ? "text-red-400" : rec.priority === 2 ? "text-yellow-400" : "text-blue-400"
                            }`}
                          >
                            {rec.priority}
                          </span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-white">{rec.title}</h3>
                            <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 text-gray-400">
                              {rec.category}
                            </span>
                          </div>
                          <p className="text-sm text-gray-400 mb-3">{rec.description}</p>
                          {rec.practice_drill && (
                            <div className="rounded-lg p-3 border border-white/5 bg-white/[0.02]">
                              <div className="flex items-center gap-2 mb-1">
                                <Zap className="w-3 h-3 text-yellow-400" />
                                <span className="text-xs font-medium text-yellow-400 uppercase">Practice Drill</span>
                              </div>
                              <p className="text-xs text-gray-400">{rec.practice_drill}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pro Comparison Radar */}
            {radarData.length > 0 && (
              <div>
                <SectionHeader
                  icon={Trophy}
                  title="Pro Player Comparison"
                  subtitle="How you compare to professional players"
                />
                <div className="gradient-card rounded-xl border border-white/5 p-6">
                  <ResponsiveContainer width="100%" height={350}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.06)" />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} />
                      <Radar name="You" dataKey="player" stroke="#ff4655" fill="#ff4655" fillOpacity={0.2} strokeWidth={2} />
                      <Radar name="nAts" dataKey="nAts" stroke="#22c55e" fill="#22c55e" fillOpacity={0.05} strokeWidth={1} strokeDasharray="4 4" />
                      <Radar name="TenZ" dataKey="TenZ" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.05} strokeWidth={1} strokeDasharray="4 4" />
                      <Legend wrapperStyle={{ fontSize: 12, color: "#9ca3af" }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Heatmap */}
            {analysis.heatmap_data && (
              <div>
                <SectionHeader
                  icon={Target}
                  title="Crosshair Heatmap"
                  subtitle="Where your crosshair spends the most time"
                />
                <div className="gradient-card rounded-xl border border-white/5 p-6">
                  <CrosshairHeatmap data={analysis.heatmap_data} />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── CROSSHAIR TAB ─── */}
        {activeTab === "crosshair" && crosshair && (
          <div className="space-y-8">
            <SectionHeader
              icon={Crosshair}
              title="Crosshair Placement Analysis"
              subtitle="The #1 factor separating pros from average players"
              score={analysis.crosshair_score || 0}
              weight="60% of score"
            />

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                icon={Target}
                label="Head Level"
                value={crosshair.head_level_consistency.toFixed(1)}
                unit="%"
                status={crosshair.head_level_consistency >= 80 ? "good" : crosshair.head_level_consistency >= 60 ? "warn" : "bad"}
              />
              <StatCard
                icon={AlertTriangle}
                label="Floor Aiming"
                value={crosshair.floor_aiming_percentage.toFixed(1)}
                unit="%"
                status={crosshair.floor_aiming_percentage <= 5 ? "good" : crosshair.floor_aiming_percentage <= 15 ? "warn" : "bad"}
              />
              <StatCard
                icon={Eye}
                label="Edge Aiming"
                value={crosshair.center_vs_edge_ratio.toFixed(1)}
                unit="%"
                status={crosshair.center_vs_edge_ratio >= 70 ? "good" : crosshair.center_vs_edge_ratio >= 50 ? "warn" : "bad"}
              />
              <StatCard
                icon={Zap}
                label="Contact Efficiency"
                value={crosshair.first_contact_efficiency.toFixed(1)}
                unit="px"
                status={crosshair.first_contact_efficiency <= 5 ? "good" : crosshair.first_contact_efficiency <= 15 ? "warn" : "bad"}
              />
            </div>

            {/* Head Level Over Time */}
            {crosshairTimeData.length > 0 && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Head Level Tracking Over Time</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={crosshairTimeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" tick={{ fill: "#6b7280", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "#1f2937", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
                      labelStyle={{ color: "#9ca3af" }}
                    />
                    <Area type="stepAfter" dataKey="headLevel" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} name="At Head Level" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Heatmap */}
            {analysis.heatmap_data && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Crosshair Position Heatmap</h3>
                <CrosshairHeatmap data={analysis.heatmap_data} />
              </div>
            )}

            {/* Pro Benchmark Comparison */}
            <div className="gradient-card rounded-xl border border-white/5 p-6">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Crosshair Benchmarks</h3>
              <div className="space-y-3">
                {[
                  { label: "Your Head Level", value: crosshair.head_level_consistency, color: "#ff4655" },
                  { label: "nAts Benchmark", value: 95, color: "#22c55e" },
                  { label: "TenZ Benchmark", value: 93, color: "#3b82f6" },
                  { label: "Average Player", value: 55, color: "#6b7280" },
                ].map((b) => (
                  <div key={b.label} className="flex items-center gap-3">
                    <span className="text-xs text-gray-500 w-32">{b.label}</span>
                    <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${b.value}%`, backgroundColor: b.color }} />
                    </div>
                    <span className="text-xs font-mono text-gray-400 w-10 text-right">{b.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ─── MOVEMENT TAB ─── */}
        {activeTab === "movement" && movement && (
          <div className="space-y-8">
            <SectionHeader
              icon={Activity}
              title="Movement Analysis"
              subtitle="Counter-strafing, peeking, and spray control"
              score={analysis.movement_score || 0}
              weight="20% of score"
            />

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                icon={Zap}
                label="Counter-Strafe"
                value={movement.counter_strafe_accuracy.toFixed(1)}
                unit="%"
                status={movement.counter_strafe_accuracy >= 80 ? "good" : movement.counter_strafe_accuracy >= 60 ? "warn" : "bad"}
              />
              <StatCard
                icon={AlertTriangle}
                label="Moving & Shooting"
                value={movement.movement_while_shooting.toFixed(1)}
                unit="%"
                status={movement.movement_while_shooting <= 10 ? "good" : movement.movement_while_shooting <= 25 ? "warn" : "bad"}
              />
              <StatCard
                icon={Shield}
                label="Spray Control"
                value={movement.spray_control_score.toFixed(0)}
                unit="/100"
                status={movement.spray_control_score >= 80 ? "good" : movement.spray_control_score >= 60 ? "warn" : "bad"}
              />
              <StatCard
                icon={TrendingUp}
                label="Over-Peeking"
                value={movement.peek_type_distribution.over?.toFixed(1) || "0"}
                unit="%"
                status={(movement.peek_type_distribution.over || 0) <= 10 ? "good" : (movement.peek_type_distribution.over || 0) <= 20 ? "warn" : "bad"}
              />
            </div>

            {/* Peek Distribution */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Peek Type Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={peekData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      dataKey="value"
                      label={({ name, value }: { name: string; value: number }) => `${name}: ${value}%`}
                    >
                      {peekData.map((_entry, index) => (
                        <Cell key={index} fill={PEEK_COLORS[index % PEEK_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-2">
                  {peekData.map((d, i) => (
                    <span key={d.name} className="flex items-center gap-1 text-xs text-gray-400">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PEEK_COLORS[i] }} />
                      {d.name}
                    </span>
                  ))}
                </div>
              </div>

              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Movement Benchmarks</h3>
                <div className="space-y-4">
                  {[
                    { label: "Counter-Strafe Accuracy", value: movement.counter_strafe_accuracy, benchmark: 90, color: "#ff4655" },
                    { label: "Shooting While Still", value: 100 - movement.movement_while_shooting, benchmark: 95, color: "#3b82f6" },
                    { label: "Spray Control", value: movement.spray_control_score, benchmark: 85, color: "#eab308" },
                  ].map((b) => (
                    <div key={b.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-500">{b.label}</span>
                        <span className="text-gray-400">{b.value.toFixed(0)}% (Pro: {b.benchmark}%)</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden relative">
                        <div className="h-full rounded-full" style={{ width: `${b.value}%`, backgroundColor: b.color }} />
                        <div className="absolute top-0 bottom-0 w-0.5 bg-white/40" style={{ left: `${b.benchmark}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── DECISION TAB ─── */}
        {activeTab === "decision" && decision && (
          <div className="space-y-8">
            <SectionHeader
              icon={Brain}
              title="Decision Making Analysis"
              subtitle="Tactical positioning, trades, and utility usage"
              score={analysis.decision_score || 0}
              weight="15% of score"
            />

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                icon={Eye}
                label="Multi-Angle Exposure"
                value={decision.multi_angle_exposure_count}
                unit="times"
                status={decision.multi_angle_exposure_count <= 5 ? "good" : decision.multi_angle_exposure_count <= 15 ? "warn" : "bad"}
              />
              <StatCard
                icon={TrendingUp}
                label="Trade Efficiency"
                value={decision.trade_efficiency.toFixed(1)}
                unit="%"
                status={decision.trade_efficiency >= 70 ? "good" : decision.trade_efficiency >= 50 ? "warn" : "bad"}
              />
              <StatCard
                icon={Star}
                label="Utility Impact"
                value={decision.utility_impact_score.toFixed(0)}
                unit="/100"
                status={decision.utility_impact_score >= 70 ? "good" : decision.utility_impact_score >= 40 ? "warn" : "bad"}
              />
              <StatCard
                icon={Shield}
                label="Cover Usage"
                value={decision.commitment_clarity.toFixed(1)}
                unit="%"
                status={decision.commitment_clarity >= 60 ? "good" : decision.commitment_clarity >= 40 ? "warn" : "bad"}
              />
            </div>

            {/* Exposure Timeline */}
            {decision.exposure_timeline && decision.exposure_timeline.length > 0 && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Angle Exposure Over Time</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={decision.exposure_timeline.slice(0, 60)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="timestamp" tick={{ fill: "#6b7280", fontSize: 10 }} tickFormatter={(v: number) => `${v}s`} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} domain={[0, 4]} ticks={[1, 2, 3, 4]} />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                    <Area type="stepAfter" dataKey="angles" stroke="#f97316" fill="#f97316" fillOpacity={0.1} name="Exposed Angles" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Utility Events */}
            {decision.utility_events && decision.utility_events.length > 0 && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Utility Usage Timeline</h3>
                <div className="flex flex-wrap gap-2">
                  {decision.utility_events.map((e, i) => (
                    <div
                      key={i}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${
                        e.type === "smoke"
                          ? "bg-blue-500/10 border-blue-500/20 text-blue-400"
                          : e.type === "flash"
                          ? "bg-yellow-500/10 border-yellow-500/20 text-yellow-400"
                          : "bg-orange-500/10 border-orange-500/20 text-orange-400"
                      }`}
                    >
                      {e.type.charAt(0).toUpperCase() + e.type.slice(1)} @ {e.timestamp.toFixed(0)}s
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── MAP TAB ─── */}
        {activeTab === "map" && (
          <div className="space-y-8">
            <SectionHeader
              icon={MapPin}
              title="Análise de Mapa/Posicionamento"
              subtitle="Rotação, zonas e exposição"
              score={analysis.map_score || 0}
              weight="10% of score"
            />

            {!mapData ? (
              <div className="gradient-card rounded-xl border border-white/5 p-6 text-gray-400">
                Não foi possível extrair o minimapa deste vídeo.
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatCard
                    icon={MapPin}
                    label="Score de Posicionamento"
                    value={mapData.positioning_score.toFixed(1)}
                    unit="/100"
                    status={mapData.positioning_score >= 70 ? "good" : mapData.positioning_score >= 50 ? "warn" : "bad"}
                  />
                  <StatCard
                    icon={AlertTriangle}
                    label="Tempo Exposto"
                    value={mapData.exposed_positioning_pct.toFixed(0)}
                    unit="%"
                    status={mapData.exposed_positioning_pct <= 15 ? "good" : mapData.exposed_positioning_pct <= 30 ? "warn" : "bad"}
                  />
                  <StatCard
                    icon={Navigation}
                    label="Rotações"
                    value={mapData.rotation_count}
                    unit=""
                    status={mapData.rotation_count <= 8 ? "good" : mapData.rotation_count <= 12 ? "warn" : "bad"}
                  />
                  <StatCard
                    icon={Timer}
                    label="Tempo Médio de Rotação"
                    value={mapData.avg_rotation_time.toFixed(1)}
                    unit="s"
                    status={mapData.avg_rotation_time <= 4 ? "good" : mapData.avg_rotation_time <= 6 ? "warn" : "bad"}
                  />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="gradient-card rounded-xl border border-white/5 p-6">
                    <h3 className="text-sm font-medium text-gray-400 mb-4">Tempo por Zona</h3>
                    {zonePieData.length > 0 ? (
                      <>
                        <ResponsiveContainer width="100%" height={260}>
                          <PieChart>
                            <Pie
                              data={zonePieData}
                              dataKey="value"
                              nameKey="name"
                              cx="50%"
                              cy="50%"
                              innerRadius={60}
                              outerRadius={95}
                              label={({ name, value }: { name: string; value: number }) => `${name}: ${value}%`}
                            >
                              {zonePieData.map((d, i) => (
                                <Cell key={i} fill={d.fill} />
                              ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="flex flex-wrap justify-center gap-3 mt-2">
                          {zonePieData.map((d) => (
                            <span key={d.name} className="flex items-center gap-2 text-xs text-gray-400">
                              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.fill }} />
                              {d.name}
                            </span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="text-sm text-gray-500">Sem dados de zona.</p>
                    )}
                  </div>

                  <div className="gradient-card rounded-xl border border-white/5 p-6">
                    <h3 className="text-sm font-medium text-gray-400 mb-4">Eventos de Posicionamento</h3>
                    {mapData.positioning_events && mapData.positioning_events.length > 0 ? (
                      <div className="space-y-3">
                        {mapData.positioning_events.slice(0, 12).map((e, i) => (
                          <div
                            key={i}
                            className={`p-3 rounded-lg border ${
                              e.severity === "high"
                                ? "bg-red-500/5 border-red-500/15"
                                : e.severity === "medium"
                                ? "bg-yellow-500/5 border-yellow-500/15"
                                : "bg-green-500/5 border-green-500/15"
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm text-gray-200">{e.description}</span>
                              <span className="text-xs font-mono text-gray-500">{e.timestamp.toFixed(0)}s</span>
                            </div>
                            <div className="mt-1 text-xs text-gray-500">{e.event_type}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">Sem eventos detectados.</p>
                    )}
                  </div>
                </div>

                <div className="gradient-card rounded-xl border border-white/5 p-6">
                  <h3 className="text-sm font-medium text-gray-400 mb-4">Timeline de Zonas</h3>
                  {mapData.zone_timeline && mapData.zone_timeline.length > 0 ? (
                    <div className="space-y-2">
                      {mapData.zone_timeline.slice(0, 18).map((z, i) => (
                        <div key={i} className="flex items-center gap-3 text-sm">
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: zoneColors[z.zone] || "#6b7280" }}
                          />
                          <span className="text-gray-300 w-28">{zoneLabels[z.zone] || z.zone}</span>
                          <span className="text-gray-500 flex-1">{z.duration.toFixed(1)}s</span>
                          <span className="text-xs font-mono text-gray-600">t={z.timestamp.toFixed(0)}s</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">Sem timeline.</p>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* ─── COMMS TAB ─── */}
        {activeTab === "comms" && comms && (
          <div className="space-y-8">
            <SectionHeader
              icon={MessageSquare}
              title="Análise de Comunicação"
              subtitle="Qualidade e timing das callouts"
              score={analysis.communication_score || 0}
              weight="5% of score"
            />

            <div className="grid grid-cols-3 gap-4">
              <StatCard
                icon={MessageSquare}
                label="Total de Callouts"
                value={comms.total_callouts}
                status={comms.total_callouts >= 10 ? "good" : comms.total_callouts >= 5 ? "warn" : "bad"}
              />
              <StatCard
                icon={Clock}
                label="Callouts no Tempo"
                value={comms.timely_callouts_pct.toFixed(0)}
                unit="%"
                status={comms.timely_callouts_pct >= 80 ? "good" : comms.timely_callouts_pct >= 60 ? "warn" : "bad"}
              />
              <StatCard
                icon={AlertTriangle}
                label="Callouts Atrasadas"
                value={comms.late_callouts_pct.toFixed(0)}
                unit="%"
                status={comms.late_callouts_pct <= 15 ? "good" : comms.late_callouts_pct <= 30 ? "warn" : "bad"}
              />
            </div>

            {/* Transcription */}
            {comms.transcription_segments && comms.transcription_segments.length > 0 && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Transcrição de Callouts</h3>
                <div className="space-y-3">
                  {comms.transcription_segments.map((seg, i) => (
                    <div
                      key={i}
                      className={`flex items-start gap-3 p-3 rounded-lg border ${
                        seg.is_timely ? "bg-green-500/5 border-green-500/10" : "bg-red-500/5 border-red-500/10"
                      }`}
                    >
                      <div className="text-xs font-mono text-gray-500 w-20 shrink-0 pt-0.5">
                        {seg.start.toFixed(1)}s - {seg.end.toFixed(1)}s
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-gray-200">&ldquo;{seg.text}&rdquo;</p>
                        <div className="flex items-center gap-2 mt-1">
                          {seg.is_callout && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">Callout</span>
                          )}
                          <span
                            className={`text-xs px-1.5 py-0.5 rounded ${
                              seg.is_timely ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
                            }`}
                          >
                            {seg.is_timely ? "No tempo" : "Atrasada"}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── ROUNDS TAB ─── */}
        {activeTab === "rounds" && (
          <div className="space-y-8">
            <SectionHeader
              icon={Clock}
              title="Análise Round por Round"
              subtitle="Desempenho detalhado por round"
            />

            {/* Round Scores Chart */}
            {roundChartData.length > 0 && (
              <div className="gradient-card rounded-xl border border-white/5 p-6">
                <h3 className="text-sm font-medium text-gray-400 mb-4">Score por Round</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={roundChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="round" tick={{ fill: "#6b7280", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} domain={[0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: "#1f2937", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                    <Bar dataKey="crosshair" fill="#ff4655" name="Crosshair" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="movement" fill="#3b82f6" name="Movement" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="decision" fill="#eab308" name="Decision" radius={[2, 2, 0, 0]} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#9ca3af" }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Round Details */}
            {analysis.round_analysis && (
              <div className="space-y-3">
                {analysis.round_analysis.map((round) => (
                  <div
                    key={round.round_number}
                    className={`gradient-card rounded-xl border p-4 ${
                      round.outcome === "win" ? "border-green-500/10" : "border-red-500/10"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${
                            round.outcome === "win" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                          }`}
                        >
                          {round.round_number}
                        </div>
                        <div>
                          <span className={`text-sm font-medium ${round.outcome === "win" ? "text-green-400" : "text-red-400"}`}>
                            {round.outcome === "win" ? "Round Vencido" : "Round Perdido"}
                          </span>
                          <p className="text-xs text-gray-500">{round.notes}</p>
                        </div>
                      </div>
                      <div className="flex gap-4">
                        <div className="text-center">
                          <div className="text-sm font-bold text-red-400">{round.crosshair_score.toFixed(0)}</div>
                          <div className="text-xs text-gray-600">XHR</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-bold text-blue-400">{round.movement_score.toFixed(0)}</div>
                          <div className="text-xs text-gray-600">MOV</div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-bold text-yellow-400">{round.decision_score.toFixed(0)}</div>
                          <div className="text-xs text-gray-600">DEC</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Navigation Bar ───
function NavBar({
  currentPage,
  onNavigate,
}: {
  currentPage: AppPage;
  onNavigate: (page: AppPage) => void;
}) {
  const navItems: { page: AppPage; label: string; icon: React.ElementType }[] = [
    { page: "analysis", label: "Análise de VOD", icon: Crosshair },
    { page: "datasets", label: "Datasets", icon: Activity },
    { page: "knowledge", label: "Knowledge Base", icon: Brain },
  ];

  return (
    <nav className="sticky top-0 z-40 bg-gray-950/80 backdrop-blur-lg border-b border-white/5">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex items-center h-14 gap-8">
          {/* Logo */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="w-8 h-8 rounded-lg gradient-valorant flex items-center justify-center">
              <Crosshair className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm font-bold text-white hidden sm:block">
              Valorant <span className="text-red-500">AI</span>
            </span>
          </div>

          {/* Nav items */}
          <div className="flex items-center gap-1">
            {navItems.map(({ page, label, icon: Icon }) => (
              <button
                key={page}
                onClick={() => onNavigate(page)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
                  ${currentPage === page
                    ? "bg-white/10 text-white"
                    : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

// ─── Main App ───
function App() {
  const [page, setPage] = useState<AppPage>("analysis");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleLoadDemo = useCallback(async () => {
    setLoading(true);
    try {
      const demo = await getDemoAnalysis();
      setAnalysis(demo);
    } catch (err) {
      console.error("Failed to load demo:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNavigate = useCallback((newPage: AppPage) => {
    setPage(newPage);
    if (newPage !== "analysis") {
      setAnalysis(null);
    }
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 text-red-400 animate-spin" />
          <p className="text-gray-400">Loading analysis...</p>
        </div>
      </div>
    );
  }

  // Dashboard view (no nav bar - has its own back button)
  if (page === "analysis" && analysis) {
    return <Dashboard analysis={analysis} onBack={() => setAnalysis(null)} />;
  }

  return (
    <div className="min-h-screen">
      <NavBar currentPage={page} onNavigate={handleNavigate} />
      {page === "analysis" && (
        <UploadSection onAnalysisReady={setAnalysis} onLoadDemo={handleLoadDemo} />
      )}
      {page === "datasets" && <DatasetManager />}
      {page === "knowledge" && <KnowledgeBase />}
    </div>
  );
}

export default App;
