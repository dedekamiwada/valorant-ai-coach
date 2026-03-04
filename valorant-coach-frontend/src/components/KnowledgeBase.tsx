import { useState, useEffect, useCallback } from "react";
import {
  Brain,
  Trash2,
  Loader2,
  AlertTriangle,
  X,
  Plus,
  Zap,
  Target,
  Activity,
  MapPin,
  TrendingUp,
  Filter,
  BarChart3,
  Shield,
} from "lucide-react";
import type { KnowledgeListItem } from "../types/knowledge";
import type { KnowledgeStats, KnowledgeCreate } from "../types/knowledge";
import {
  listKnowledge,
  deleteKnowledgeEntry,
  getKnowledgeStats,
  createKnowledgeEntry,
  extractKnowledgeFromAnalysis,
} from "../api/knowledge";

const CATEGORIES = [
  { value: "crosshair", label: "Crosshair", icon: Target, color: "text-green-400" },
  { value: "movement", label: "Movimento", icon: Activity, color: "text-blue-400" },
  { value: "positioning", label: "Posicionamento", icon: MapPin, color: "text-purple-400" },
  { value: "tactical", label: "Tático", icon: Shield, color: "text-yellow-400" },
  { value: "ability_usage", label: "Habilidades", icon: Zap, color: "text-orange-400" },
  { value: "benchmark", label: "Benchmark", icon: TrendingUp, color: "text-cyan-400" },
  { value: "economy", label: "Economia", icon: BarChart3, color: "text-emerald-400" },
];

function getCategoryInfo(category: string) {
  return CATEGORIES.find((c) => c.value === category) || CATEGORIES[0];
}

function confidenceBar(confidence: number) {
  const pct = Math.round(confidence * 100);
  const color =
    confidence >= 0.8
      ? "bg-green-500"
      : confidence >= 0.6
      ? "bg-yellow-500"
      : confidence >= 0.4
      ? "bg-orange-500"
      : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500">{pct}%</span>
    </div>
  );
}

export default function KnowledgeBase() {
  const [entries, setEntries] = useState<KnowledgeListItem[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [filterCategory, setFilterCategory] = useState("");
  const [filterSource, setFilterSource] = useState("");

  // Add entry form
  const [showAddForm, setShowAddForm] = useState(false);
  const [formCategory, setFormCategory] = useState("crosshair");
  const [formTitle, setFormTitle] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formAgent, setFormAgent] = useState("");
  const [formMap, setFormMap] = useState("");
  const [adding, setAdding] = useState(false);

  // Extract from analysis
  const [showExtractForm, setShowExtractForm] = useState(false);
  const [extractAnalysisId, setExtractAnalysisId] = useState("");
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<string | null>(null);

  // Delete
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const params: { category?: string; source_type?: string } = {};
      if (filterCategory) params.category = filterCategory;
      if (filterSource) params.source_type = filterSource;

      const [entryList, entryStats] = await Promise.all([
        listKnowledge(params),
        getKnowledgeStats(),
      ]);
      setEntries(entryList);
      setStats(entryStats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao carregar knowledge base");
    } finally {
      setLoading(false);
    }
  }, [filterCategory, filterSource]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async () => {
    if (!formTitle || !formDescription) return;
    setAdding(true);
    setError(null);
    try {
      const entry: KnowledgeCreate = {
        source_type: "manual",
        category: formCategory,
        title: formTitle,
        description: formDescription,
        agent: formAgent || undefined,
        map_name: formMap || undefined,
        confidence: 0.5,
      };
      await createKnowledgeEntry(entry);
      setShowAddForm(false);
      setFormTitle("");
      setFormDescription("");
      setFormAgent("");
      setFormMap("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao criar entry");
    } finally {
      setAdding(false);
    }
  };

  const handleExtract = async () => {
    if (!extractAnalysisId) return;
    setExtracting(true);
    setError(null);
    setExtractResult(null);
    try {
      const result = await extractKnowledgeFromAnalysis(extractAnalysisId);
      setExtractResult(`${result.entries_created} insights extraídos da análise.`);
      setExtractAnalysisId("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha ao extrair knowledge");
    } finally {
      setExtracting(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteKnowledgeEntry(id);
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
          <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center">
            <Brain className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Knowledge Base</h1>
            <p className="text-sm text-gray-500">Memória e aprendizado acumulado da AI</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowExtractForm(true)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 font-medium hover:bg-purple-500/20 transition-colors text-sm"
          >
            <Zap className="w-4 h-4" />
            Extrair de Análise
          </button>
          <button
            onClick={() => setShowAddForm(true)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl gradient-valorant text-white font-medium hover:opacity-90 transition-opacity text-sm"
          >
            <Plus className="w-4 h-4" />
            Adicionar Insight
          </button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <Brain className="w-4 h-4 text-purple-400" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Total Insights</span>
            </div>
            <span className="text-2xl font-bold text-white">{stats.total_entries}</span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Confiança Média</span>
            </div>
            <span className="text-2xl font-bold text-green-400">
              {(stats.avg_confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <BarChart3 className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Categorias</span>
            </div>
            <span className="text-2xl font-bold text-blue-400">
              {Object.keys(stats.categories).length}
            </span>
          </div>
          <div className="gradient-card rounded-xl border border-white/5 p-4">
            <div className="flex items-center gap-2 mb-1">
              <MapPin className="w-4 h-4 text-yellow-400" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">Mapas</span>
            </div>
            <span className="text-2xl font-bold text-yellow-400">
              {Object.keys(stats.maps).length}
            </span>
          </div>
        </div>
      )}

      {/* Category breakdown chips */}
      {stats && Object.keys(stats.categories).length > 0 && (
        <div className="flex flex-wrap gap-2 mb-6">
          {Object.entries(stats.categories).map(([cat, count]) => {
            const info = getCategoryInfo(cat);
            const Icon = info.icon;
            return (
              <div
                key={cat}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-800/50 border border-white/5 text-xs"
              >
                <Icon className={`w-3 h-3 ${info.color}`} />
                <span className="text-gray-300 capitalize">{info.label}</span>
                <span className="text-gray-600">({count})</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="flex items-center gap-2 text-gray-500">
          <Filter className="w-4 h-4" />
          <span className="text-sm">Filtros:</span>
        </div>
        <select
          value={filterCategory}
          onChange={(e) => setFilterCategory(e.target.value)}
          className="bg-gray-800/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-white/20"
        >
          <option value="">Todas categorias</option>
          {CATEGORIES.map((c) => (
            <option key={c.value} value={c.value}>{c.label}</option>
          ))}
        </select>
        <select
          value={filterSource}
          onChange={(e) => setFilterSource(e.target.value)}
          className="bg-gray-800/50 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-gray-300 focus:outline-none focus:border-white/20"
        >
          <option value="">Todas fontes</option>
          <option value="analysis">Auto-extraído</option>
          <option value="manual">Manual</option>
          <option value="pro_benchmark">Pro Benchmark</option>
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

      {/* Extract Result */}
      {extractResult && (
        <div className="mb-6 p-4 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center gap-3">
          <Zap className="w-5 h-5 text-green-400 shrink-0" />
          <p className="text-green-300 text-sm">{extractResult}</p>
          <button onClick={() => setExtractResult(null)} className="ml-auto">
            <X className="w-4 h-4 text-green-400" />
          </button>
        </div>
      )}

      {/* Add Entry Modal */}
      {showAddForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Adicionar Insight</h2>
              <button onClick={() => setShowAddForm(false)} disabled={adding}>
                <X className="w-5 h-5 text-gray-400 hover:text-white" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Categoria *</label>
                <select
                  value={formCategory}
                  onChange={(e) => setFormCategory(e.target.value)}
                  className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/20"
                >
                  {CATEGORIES.map((c) => (
                    <option key={c.value} value={c.value}>{c.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Título *</label>
                <input
                  type="text"
                  value={formTitle}
                  onChange={(e) => setFormTitle(e.target.value)}
                  placeholder="Ex: Counter-strafe melhora hit rate em 40%"
                  className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Descrição *</label>
                <textarea
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  rows={3}
                  placeholder="Descreva o insight tático em detalhes..."
                  className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20 resize-none"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Agente</label>
                  <input
                    type="text"
                    value={formAgent}
                    onChange={(e) => setFormAgent(e.target.value)}
                    placeholder="Ex: Jett"
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider mb-1 block">Mapa</label>
                  <input
                    type="text"
                    value={formMap}
                    onChange={(e) => setFormMap(e.target.value)}
                    placeholder="Ex: Ascent"
                    className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowAddForm(false)}
                disabled={adding}
                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white border border-white/10"
              >
                Cancelar
              </button>
              <button
                onClick={handleAdd}
                disabled={!formTitle || !formDescription || adding}
                className="px-5 py-2 rounded-lg text-sm font-medium text-white gradient-valorant hover:opacity-90 disabled:opacity-40 transition-all"
              >
                {adding ? "Salvando..." : "Salvar"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Extract from Analysis Modal */}
      {showExtractForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white">Extrair Knowledge de Análise</h2>
              <button onClick={() => setShowExtractForm(false)} disabled={extracting}>
                <X className="w-5 h-5 text-gray-400 hover:text-white" />
              </button>
            </div>

            <p className="text-sm text-gray-400 mb-4">
              Cole o ID de uma análise concluída para extrair automaticamente insights
              de crosshair, movimento, decisão e posicionamento.
            </p>

            <input
              type="text"
              value={extractAnalysisId}
              onChange={(e) => setExtractAnalysisId(e.target.value)}
              placeholder="ID da análise (ex: abc123-def456...)"
              className="w-full bg-gray-800/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-white/20"
            />

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowExtractForm(false)}
                disabled={extracting}
                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white border border-white/10"
              >
                Cancelar
              </button>
              <button
                onClick={handleExtract}
                disabled={!extractAnalysisId || extracting}
                className="px-5 py-2 rounded-lg text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 disabled:opacity-40 transition-all inline-flex items-center gap-2"
              >
                {extracting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Extraindo...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Extrair
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {deletingId && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/10 rounded-2xl w-full max-w-sm p-6">
            <h3 className="text-lg font-semibold text-white mb-2">Deletar Insight?</h3>
            <p className="text-sm text-gray-400 mb-6">
              Esta ação não pode ser desfeita.
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

      {/* Entry List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-purple-400 animate-spin" />
        </div>
      ) : entries.length === 0 ? (
        <div className="text-center py-20">
          <Brain className="w-16 h-16 text-gray-700 mx-auto mb-4" />
          <p className="text-lg text-gray-400 mb-2">Nenhum insight ainda</p>
          <p className="text-sm text-gray-600">
            Analise VODs ou adicione insights manualmente para construir a base de conhecimento
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {entries.map((e) => {
            const catInfo = getCategoryInfo(e.category);
            const CatIcon = catInfo.icon;
            return (
              <div
                key={e.id}
                className="gradient-card rounded-xl border border-white/5 hover:border-white/10 transition-colors p-4"
              >
                <div className="flex items-start gap-4">
                  {/* Category icon */}
                  <div className="w-10 h-10 rounded-lg bg-gray-800/50 flex items-center justify-center shrink-0">
                    <CatIcon className={`w-5 h-5 ${catInfo.color}`} />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm font-semibold text-white">{e.title}</h3>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className={`px-2 py-0.5 rounded-full bg-gray-800/50 ${catInfo.color}`}>
                        {catInfo.label}
                      </span>
                      {e.subcategory && (
                        <span className="px-2 py-0.5 rounded-full bg-gray-800/50 text-gray-400">
                          {e.subcategory}
                        </span>
                      )}
                      {e.agent && (
                        <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400">
                          {e.agent}
                        </span>
                      )}
                      {e.map_name && (
                        <span className="px-2 py-0.5 rounded-full bg-green-500/10 text-green-400">
                          {e.map_name}
                        </span>
                      )}
                      <span className="px-2 py-0.5 rounded-full bg-gray-800/50 text-gray-500">
                        {e.source_type}
                      </span>
                      <span className="text-gray-600">
                        {e.observation_count}x observado
                      </span>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="shrink-0">
                    {confidenceBar(e.confidence)}
                  </div>

                  {/* Delete */}
                  <button
                    onClick={() => setDeletingId(e.id)}
                    className="text-gray-600 hover:text-red-400 transition-colors p-1 shrink-0"
                    title="Deletar"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
