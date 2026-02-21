import { ExternalLink, Heart } from 'lucide-react'
import { useSession } from '../context/SessionContext'

const merchantColors = {
  arcaplanet: 'bg-green-100 text-green-800',
  twinset: 'bg-purple-100 text-purple-800',
}

export default function ProductCard({ product }) {
  const { name, price, currency, merchant, image_url, url, score, product_id } = product
  const badgeClass = merchantColors[merchant] || 'bg-gray-100 text-gray-800'
  const { isFavorite, toggleFavorite } = useSession()
  const fav = isFavorite(product_id)

  return (
    <div className="relative flex-shrink-0 w-52 rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* Heart button */}
      <button
        onClick={(e) => { e.stopPropagation(); toggleFavorite(product_id) }}
        className="absolute top-2 right-2 z-10 p-1 rounded-full bg-white/80 hover:bg-white transition-colors"
        title={fav ? 'Remove from favorites' : 'Add to favorites'}
      >
        <Heart
          size={18}
          className={fav ? 'fill-red-500 text-red-500' : 'text-gray-400 hover:text-red-400'}
        />
      </button>

      <div className="h-44 bg-gray-50 flex items-center justify-center overflow-hidden">
        {image_url ? (
          <img
            src={image_url}
            alt={name}
            className="h-full w-full object-contain"
            loading="lazy"
          />
        ) : (
          <span className="text-gray-400 text-sm">No image</span>
        )}
      </div>
      <div className="p-3 space-y-2">
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${badgeClass}`}>
          {merchant}
        </span>
        <p className="text-sm font-medium text-gray-900 line-clamp-2 leading-tight" title={name}>
          {name}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-gray-900">
            {price ? `${price} ${currency || 'EUR'}` : 'N/A'}
          </span>
          {score != null && (
            <span className="text-xs text-gray-400" title="Similarity score">
              {(score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-1 w-full py-1.5 rounded-md text-xs font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 transition-colors"
        >
          Visit Site <ExternalLink size={12} />
        </a>
      </div>
    </div>
  )
}
