import { useState, useEffect } from 'react'
import { fetchProducts } from '../api'
import { useSession } from '../context/SessionContext'
import ProductCard from '../components/ProductCard'
import { Loader2, Heart } from 'lucide-react'

export default function BrowsePage() {
  const [products, setProducts] = useState([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [merchant, setMerchant] = useState('')
  const [loading, setLoading] = useState(true)
  const [showFavs, setShowFavs] = useState(false)
  const [favProducts, setFavProducts] = useState([])
  const { sessionId } = useSession()
  const limit = 20

  useEffect(() => {
    if (showFavs) {
      setLoading(true)
      fetch(`/api/favorites?session_id=${sessionId}`)
        .then(r => r.json())
        .then(data => {
          let items = data.favorites || []
          if (merchant) items = items.filter(p => p.merchant === merchant)
          setFavProducts(items)
          setTotal(items.length)
        })
        .catch(console.error)
        .finally(() => setLoading(false))
    } else {
      setLoading(true)
      fetchProducts({ merchant: merchant || undefined, offset, limit })
        .then((data) => {
          setProducts(data.products)
          setTotal(data.total)
        })
        .catch(console.error)
        .finally(() => setLoading(false))
    }
  }, [merchant, offset, showFavs, sessionId])

  const displayProducts = showFavs ? favProducts : products

  return (
    <div className="max-w-5xl mx-auto px-4 py-6">
      <div className="flex items-center gap-4 mb-6">
        <h1 className="text-xl font-semibold">Browse Catalog</h1>
        <select
          value={merchant}
          onChange={(e) => { setMerchant(e.target.value); setOffset(0) }}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
        >
          <option value="">All Merchants</option>
          <option value="arcaplanet">Arcaplanet</option>
          <option value="twinset">Twinset</option>
        </select>
        <button
          onClick={() => { setShowFavs(!showFavs); setOffset(0) }}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            showFavs
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'border border-gray-300 text-gray-600 hover:bg-gray-50'
          }`}
        >
          <Heart size={14} className={showFavs ? 'fill-red-500 text-red-500' : ''} />
          Favorites
        </button>
        <span className="text-sm text-gray-500">{total} products</span>
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader2 size={24} className="animate-spin text-gray-400" />
        </div>
      ) : displayProducts.length === 0 ? (
        <div className="text-center py-20 text-gray-400">
          {showFavs ? 'No favorites yet. Click the heart on any product to save it.' : 'No products found.'}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {displayProducts.map((p) => (
              <ProductCard key={p.product_id} product={p} />
            ))}
          </div>
          {!showFavs && (
            <div className="flex justify-center gap-4 mt-8">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="px-4 py-2 rounded-lg border border-gray-300 text-sm disabled:opacity-50"
              >
                Previous
              </button>
              <span className="py-2 text-sm text-gray-500">
                {offset + 1}–{Math.min(offset + limit, total)} of {total}
              </span>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
                className="px-4 py-2 rounded-lg border border-gray-300 text-sm disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
