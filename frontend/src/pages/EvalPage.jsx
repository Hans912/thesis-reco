import { useState, useEffect } from 'react'
import { Loader2, BarChart3, ShoppingBag, Store } from 'lucide-react'
import { fetchEvaluationResults } from '../api'

const K_OPTIONS = [3, 5, 10, 15, 20]

// Metrics shown per level — novelty is product-only
const STORE_METRICS = [
  { key: 'hit_rate',  label: 'Hit Rate@K' },
  { key: 'ndcg',     label: 'nDCG@K' },
  { key: 'precision', label: 'Precision@K' },
  { key: 'recall',   label: 'Recall@K' },
  { key: 'coverage', label: 'Coverage' },
  { key: 'diversity', label: 'Diversity (ILD)' },
]

const PRODUCT_METRICS = [
  { key: 'hit_rate',  label: 'Hit Rate@K' },
  { key: 'ndcg',     label: 'nDCG@K' },
  { key: 'precision', label: 'Precision@K' },
  { key: 'recall',   label: 'Recall@K' },
  { key: 'coverage', label: 'Coverage' },
  { key: 'diversity', label: 'Diversity (ILD)' },
  { key: 'novelty',  label: 'Novelty' },
]

const MODEL_COLORS = {
  // Product-level
  'Content-Based (CLIP)':         'bg-blue-500',
  'Random Baseline (Product)':    'bg-gray-400',
  'Popularity Baseline (Product)':'bg-orange-400',
  // Store — memory-based
  'Item-Based CF':                'bg-green-500',
  'User-Based CF':                'bg-purple-500',
  // Store — model-based
  'ALS':                          'bg-cyan-500',
  'LightFM WARP':                 'bg-rose-500',
  'LightFM Hybrid':               'bg-amber-500',
  // Store — baselines
  'Random Baseline':              'bg-gray-400',
  'Popularity Baseline':          'bg-orange-400',
  // Store — demographic
  'Demographic Popularity':       'bg-teal-500',
  'LightFM Demo':                 'bg-violet-500',
  'LightFM Full Hybrid':          'bg-pink-500',
}

// ── Small components ───────────────────────────────────────────────────

function StatCard({ label, value }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-4 py-3">
      <div className="text-lg font-bold text-stamp-900">{value}</div>
      <div className="text-xs text-stamp-400">{label}</div>
    </div>
  )
}

function DataStatsSection({ stats }) {
  if (!stats) return null
  const fmtPct = (v) => `${(v * 100).toFixed(2)}%`
  const fmtNum = (v) => v?.toLocaleString() ?? '—'
  return (
    <div className="mb-8 space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Customers (train)" value={fmtNum(stats.n_customers_train)} />
        <StatCard label="Stores (train)"    value={fmtNum(stats.n_stores_train)} />
        <StatCard label="Products (CLIP)"   value={fmtNum(stats.n_embeddings)} />
        <StatCard label="Transactions (train)" value={fmtNum(stats.n_train_transactions)} />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="CS Sparsity"         value={fmtPct(stats.sparsity_customer_store)} />
        <StatCard label="SP Sparsity"         value={fmtPct(stats.sparsity_store_product)} />
        <StatCard label="Avg stores/customer" value={stats.avg_stores_per_customer} />
        <StatCard label="Avg products/store"  value={stats.avg_products_per_store} />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <StatCard label="Split Date"    value={stats.split_date} />
        <StatCard label="Test Cases"    value={fmtNum(stats.n_test_cases)} />
        <StatCard label="Embedding Dim" value={stats.embedding_dim} />
      </div>
    </div>
  )
}

function MetricBar({ value, maxValue, color }) {
  const width = maxValue > 0 ? (value / maxValue) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-3.5 overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${Math.max(width, 0.5)}%` }}
        />
      </div>
      <span className="text-xs font-mono text-stamp-500 w-12 text-right">
        {value.toFixed(3)}
      </span>
    </div>
  )
}

// ── Per-level section ──────────────────────────────────────────────────

function ModelSection({ title, icon, description, models, metrics }) {
  if (!models.length) return null

  // Max per metric within this group only (so bars are relative to level peers)
  const maxPerMetric = {}
  for (const { key } of metrics) {
    maxPerMetric[key] = Math.max(...models.map(m => m.metrics[key] ?? 0), 0.001)
  }

  // Bold the best value in each metric column
  const bestPerMetric = {}
  for (const { key } of metrics) {
    bestPerMetric[key] = Math.max(...models.map(m => m.metrics[key] ?? 0))
  }

  return (
    <div className="mb-10">
      {/* Section header */}
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <div>
          <h2 className="text-base font-semibold text-stamp-900">{title}</h2>
          <p className="text-xs text-stamp-400">{description}</p>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200 mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-4 py-3 font-semibold text-stamp-800">Model</th>
              {metrics.map(({ key, label }) => (
                <th key={key} className="text-right px-3 py-3 font-semibold text-stamp-800 text-xs whitespace-nowrap">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model, idx) => {
              const color = MODEL_COLORS[model.name] || 'bg-gray-400'
              return (
                <tr key={model.name} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                  <td className="px-4 py-2.5 font-medium text-stamp-900">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color}`} />
                      {model.name}
                    </div>
                  </td>
                  {metrics.map(({ key }) => {
                    const val = model.metrics[key] ?? 0
                    const isBest = val > 0 && val === bestPerMetric[key]
                    return (
                      <td
                        key={key}
                        className={`px-3 py-2.5 text-right font-mono text-xs ${
                          isBest ? 'font-bold text-stamp-900' : 'text-stamp-500'
                        }`}
                      >
                        {val.toFixed(4)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Bar charts — one card per metric */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metrics.map(({ key, label }) => (
          <div key={key} className="rounded-xl border border-gray-200 p-4">
            <h3 className="text-xs font-semibold text-stamp-700 uppercase tracking-wide mb-3">
              {label}
            </h3>
            <div className="space-y-2">
              {models.map(model => (
                <div key={model.name}>
                  <div className="text-xs text-stamp-400 mb-0.5 truncate">{model.name}</div>
                  <MetricBar
                    value={model.metrics[key] ?? 0}
                    maxValue={maxPerMetric[key]}
                    color={MODEL_COLORS[model.name] || 'bg-gray-400'}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────────────────

export default function EvalPage() {
  const [k, setK] = useState(5)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchEvaluationResults(k)
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [k])

  const productModels = data?.models.filter(m => m.level === 'Product') ?? []
  const storeModels   = data?.models.filter(m => m.level === 'Store')   ?? []

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Page header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <BarChart3 size={22} className="text-stamp-500" />
            Model Evaluation
          </h1>
          <p className="text-sm text-stamp-400 mt-1">
            {storeModels.length} store-level models · {productModels.length} product-level models · temporal train/test split
          </p>
        </div>

        {/* K selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-stamp-400 font-medium">K =</span>
          {K_OPTIONS.map(val => (
            <button
              key={val}
              onClick={() => setK(val)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                k === val
                  ? 'bg-stamp-600 text-white'
                  : 'bg-gray-100 text-stamp-500 hover:bg-gray-200'
              }`}
            >
              {val}
            </button>
          ))}
        </div>
      </div>

      {data && !loading && <DataStatsSection stats={data.data_stats} />}

      {loading && (
        <div className="flex flex-col items-center justify-center py-24">
          <Loader2 size={28} className="animate-spin text-gray-400 mb-3" />
          <p className="text-sm text-gray-500">Loading evaluation results…</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Failed to load evaluation: {error}
        </div>
      )}

      {data && !loading && (
        <>
          {/* ── Store-level ─────────────────────────────────── */}
          <ModelSection
            title="Store-Level Recommendations"
            icon={<Store size={18} className="text-stamp-500" />}
            description="Temporal split evaluation — predict new stores customers visit after the split date. Includes memory-based CF, model-based CF, demographic models, and baselines."
            models={storeModels}
            metrics={STORE_METRICS}
          />

          {/* Divider */}
          <hr className="border-gray-200 my-2" />

          {/* ── Product-level ───────────────────────────────── */}
          <ModelSection
            title="Product-Level Recommendations"
            icon={<ShoppingBag size={18} className="text-stamp-500" />}
            description="Self-retrieval evaluation — relevance = same product category (URL slug for Twinset, name keywords for Arcaplanet). Novelty = mean self-information of recommendation frequency."
            models={productModels}
            metrics={PRODUCT_METRICS}
          />
        </>
      )}
    </div>
  )
}
