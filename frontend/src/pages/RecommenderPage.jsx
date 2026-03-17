import { useState, useEffect } from 'react'
import { MessageSquare, MapPin } from 'lucide-react'
import CFSection from '../components/CFSection'
import ChatSection from '../components/ChatSection'
import { fetchCities } from '../api'

export default function RecommenderPage() {
  const [cities, setCities] = useState([])
  const [selectedCity, setSelectedCity] = useState('')

  useEffect(() => {
    fetchCities()
      .then(setCities)
      .catch(() => setCities([]))
  }, [])

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
      {/* City selector */}
      {cities.length > 0 && (
        <div className="flex items-center gap-2">
          <MapPin size={16} className="text-stamp-400" />
          <label className="text-sm text-stamp-500 font-medium">City:</label>
          <select
            value={selectedCity}
            onChange={(e) => setSelectedCity(e.target.value)}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-stamp-400"
          >
            <option value="">All cities</option>
            {cities.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {selectedCity && (
            <button
              onClick={() => setSelectedCity('')}
              className="text-xs text-gray-400 hover:text-gray-600"
            >
              Clear
            </button>
          )}
        </div>
      )}

      {/* Top: CF store recommendations */}
      <CFSection city={selectedCity} />

      {/* Bottom: Chatbot */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={18} className="text-stamp-500" />
          <h2 className="text-lg font-semibold text-stamp-900">Product Search (Content-Based)</h2>
        </div>
        <p className="text-xs text-stamp-400 mb-3">
          Ask for product recommendations using text or images. Powered by CLIP multimodal embeddings.
        </p>
        <ChatSection city={selectedCity} />
      </div>
    </div>
  )
}
