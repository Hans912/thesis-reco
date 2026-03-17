import { useState, useEffect } from 'react'
import { useSession } from '../context/SessionContext'
import { Loader2, Store, Users, ShoppingBag, MapPin, ChevronDown, ChevronUp, Info, Sparkles, UserCircle } from 'lucide-react'

function StoreCard({ store, rank }) {
  return (
    <div className="flex items-start gap-3 p-4 rounded-lg border border-gray-200 bg-white">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-stamp-100 text-stamp-700 flex items-center justify-center text-sm font-bold">
        {rank}
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-medium text-stamp-900 truncate">{store.merchant_name || store.store_id.slice(0, 8)}</div>
        {store.city && (
          <div className="flex items-center gap-1 text-xs text-stamp-400 mt-0.5">
            <MapPin size={11} />
            {store.city}
          </div>
        )}
        <div className="flex flex-wrap gap-3 mt-2 text-xs text-stamp-400">
          <span title="Similarity score" className="font-medium text-stamp-500">
            {(store.score * 100).toFixed(1)}% match
          </span>
          {store.num_products != null && (
            <span className="flex items-center gap-1">
              <ShoppingBag size={11} /> {store.num_products} products
            </span>
          )}
          {store.median_price != null && (
            <span>Median: {store.median_price.toFixed(2)} EUR</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default function CFSection({ city = '' }) {
  const { profile } = useSession()
  const [itemBased, setItemBased] = useState(null)
  const [userBased, setUserBased] = useState(null)
  const [lightfm, setLightfm] = useState(null)
  const [demographic, setDemographic] = useState(null)
  const [loading, setLoading] = useState(false)
  const [topStore, setTopStore] = useState(null)
  const [expanded, setExpanded] = useState(true)

  const isGuest = !profile

  useEffect(() => {
    if (isGuest) {
      setItemBased(null)
      setUserBased(null)
      setLightfm(null)
      setDemographic(null)
      setTopStore(null)
      return
    }

    setLoading(true)
    const cityParam = city ? `&city=${encodeURIComponent(city)}` : ''

    Promise.all([
      fetch(`/api/recommend/stores?profile_id=${profile.profile_id}&top_k=8${cityParam}`)
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
      fetch(`/api/profiles/${profile.profile_id}/top-store`)
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
      fetch(`/api/recommend/stores/lightfm?profile_id=${profile.profile_id}&top_k=8${cityParam}`)
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
      fetch(`/api/recommend/stores/demographic?profile_id=${profile.profile_id}&top_k=8${cityParam}`)
        .then(r => r.ok ? r.json() : null)
        .catch(() => null),
    ]).then(async ([ubResult, topStoreResult, lfmResult, demoResult]) => {
      setUserBased(ubResult)
      setTopStore(topStoreResult)
      setLightfm(lfmResult)
      setDemographic(demoResult)

      if (topStoreResult?.store_id) {
        try {
          const ibRes = await fetch(`/api/stores/${topStoreResult.store_id}/similar?top_k=8${cityParam}`)
          if (ibRes.ok) setItemBased(await ibRes.json())
        } catch {}
      }

      setLoading(false)
    })
  }, [profile, isGuest, city])

  if (isGuest) {
    return (
      <div className="rounded-xl border border-stamp-100 bg-stamp-50 p-6 text-center">
        <Info size={24} className="mx-auto mb-2 text-stamp-400" />
        <p className="text-sm font-medium text-stamp-700">Store recommendations require purchase history</p>
        <p className="text-xs text-stamp-500 mt-1">Select a profile (Luca, Sofia, or Maria) to see collaborative filtering recommendations.</p>
      </div>
    )
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left mb-4"
      >
        <h2 className="text-lg font-semibold text-stamp-900">Store Recommendations for {profile.name}</h2>
        {expanded ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
      </button>

      {expanded && (
        <>
          {loading ? (
            <div className="flex justify-center py-12">
              <Loader2 size={24} className="animate-spin text-gray-400" />
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* User-based CF */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Users size={16} className="text-purple-600" />
                  <h3 className="text-sm font-semibold text-stamp-800">User-Based CF</h3>
                </div>
                <p className="text-xs text-stamp-400 mb-3">
                  Stores visited by customers with similar shopping patterns.
                </p>
                {userBased && userBased.results.length > 0 ? (
                  <div className="space-y-2">
                    {userBased.results.map((s, i) => (
                      <StoreCard key={s.store_id} store={s} rank={i + 1} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-400 text-sm">
                    No user-based recommendations available.
                  </div>
                )}
              </div>

              {/* Item-based CF */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Store size={16} className="text-green-600" />
                  <h3 className="text-sm font-semibold text-stamp-800">Item-Based CF</h3>
                </div>
                <p className="text-xs text-stamp-400 mb-3">
                  Stores with a similar product mix to {topStore?.merchant_name || 'most-visited store'}
                  {topStore?.city ? ` (${topStore.city})` : ''}.
                </p>
                {itemBased && itemBased.results.length > 0 ? (
                  <div className="space-y-2">
                    {itemBased.results.map((s, i) => (
                      <StoreCard key={s.store_id} store={s} rank={i + 1} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-400 text-sm">
                    No item-based recommendations available.
                  </div>
                )}
              </div>

              {/* LightFM WARP */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles size={16} className="text-rose-600" />
                  <h3 className="text-sm font-semibold text-stamp-800">LightFM WARP</h3>
                </div>
                <p className="text-xs text-stamp-400 mb-3">
                  Learned latent factor model optimized for top-k ranking.
                </p>
                {lightfm && lightfm.results.length > 0 ? (
                  <div className="space-y-2">
                    {lightfm.results.map((s, i) => (
                      <StoreCard key={s.store_id} store={s} rank={i + 1} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-400 text-sm">
                    No LightFM recommendations available.
                  </div>
                )}
              </div>

              {/* Demographic Popularity */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <UserCircle size={16} className="text-amber-600" />
                  <h3 className="text-sm font-semibold text-stamp-800">Demographic</h3>
                </div>
                <p className="text-xs text-stamp-400 mb-3">
                  Stores popular among customers with a similar age, nationality, and travel profile.
                </p>
                {demographic && demographic.results.length > 0 ? (
                  <div className="space-y-2">
                    {demographic.results.map((s, i) => (
                      <StoreCard key={s.store_id} store={s} rank={i + 1} />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-400 text-sm">
                    No demographic recommendations available.
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
