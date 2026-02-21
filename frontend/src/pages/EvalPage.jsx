import { useState, useEffect } from 'react'
import { Loader2, BarChart3 } from 'lucide-react'
import { fetchEvaluationResults } from '../api'

const K_OPTIONS = [3, 5, 10]

const METRIC_LABELS = {
  precision: 'Precision@K',
  recall: 'Recall@K',
  ndcg: 'nDCG@K',
  hit_rate: 'Hit Rate@K',
  coverage: 'Coverage',
  diversity: 'Diversity',
}

const MODEL_COLORS = {
  'Content-Based (CLIP)': 'bg-blue-500',
  'Item-Based CF': 'bg-green-500',
  'User-Based CF': 'bg-purple-500',
  'Random Baseline': 'bg-gray-400',
  'Popularity Baseline': 'bg-orange-400',
}

function MetricBar({ value, maxValue, color }) {
  const width = maxValue > 0 ? (value / maxValue) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${Math.max(width, 1)}%` }}
        />
      </div>
      <span className="text-xs font-mono text-gray-600 w-12 text-right">
        {value.toFixed(3)}
      </span>
    </div>
  )
}

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

  const metricKeys = Object.keys(METRIC_LABELS)

  // Find max value per metric for bar scaling
  const maxPerMetric = {}
  if (data) {
    for (const key of metricKeys) {
      maxPerMetric[key] = Math.max(...data.models.map(m => m.metrics[key] || 0), 0.001)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <BarChart3 size={22} className="text-blue-600" />
            Model Evaluation
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Comparing 5 recommendation models across 6 IR metrics.
          </p>
        </div>

        {/* K selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500 font-medium">K =</span>
          {K_OPTIONS.map(val => (
            <button
              key={val}
              onClick={() => setK(val)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                k === val
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {val}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-24">
          <Loader2 size={28} className="animate-spin text-gray-400 mb-3" />
          <p className="text-sm text-gray-500">Running evaluation (this may take a moment)...</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          Failed to load evaluation: {error}
        </div>
      )}

      {data && !loading && (
        <>
          {/* Comparison table */}
          <div className="overflow-x-auto rounded-xl border border-gray-200 mb-8">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="text-left px-4 py-3 font-semibold text-gray-700">Model</th>
                  <th className="text-left px-3 py-3 font-semibold text-gray-500 text-xs">Level</th>
                  {metricKeys.map(key => (
                    <th key={key} className="text-right px-3 py-3 font-semibold text-gray-700 text-xs">
                      {METRIC_LABELS[key]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.models.map((model, idx) => {
                  const color = MODEL_COLORS[model.name] || 'bg-gray-400'
                  const dotColor = color.replace('bg-', 'bg-')
                  return (
                    <tr key={model.name} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                      <td className="px-4 py-3 font-medium text-gray-900 flex items-center gap-2">
                        <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
                        {model.name}
                      </td>
                      <td className="px-3 py-3 text-gray-500 text-xs">{model.level}</td>
                      {metricKeys.map(key => {
                        const val = model.metrics[key] ?? 0
                        const isMax = val === maxPerMetric[key] && val > 0
                        return (
                          <td key={key} className={`px-3 py-3 text-right font-mono text-xs ${isMax ? 'font-bold text-blue-700' : 'text-gray-600'}`}>
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

          {/* Visual bar charts per metric */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {metricKeys.map(key => (
              <div key={key} className="rounded-xl border border-gray-200 p-4">
                <h3 className="text-sm font-semibold text-gray-700 mb-3">{METRIC_LABELS[key]}</h3>
                <div className="space-y-2">
                  {data.models.map(model => (
                    <div key={model.name}>
                      <div className="text-xs text-gray-500 mb-0.5">{model.name}</div>
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
        </>
      )}
    </div>
  )
}
