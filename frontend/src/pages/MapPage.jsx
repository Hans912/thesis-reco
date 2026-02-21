import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { fetchStores } from '../api'
import { Loader2 } from 'lucide-react'

// Fix default marker icons (Leaflet + bundler issue)
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const greenIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

const violetIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-violet.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

const merchantIcon = {
  arcaplanet: greenIcon,
  twinset: violetIcon,
}

export default function MapPage() {
  const [stores, setStores] = useState([])
  const [merchant, setMerchant] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchStores({ merchant: merchant || undefined })
      .then((data) => setStores(data.stores))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [merchant])

  // Center on Italy by default
  const center = stores.length > 0
    ? [stores[0].lat, stores[0].lng]
    : [42.5, 12.5]

  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">
      <div className="px-4 py-3 border-b border-gray-200 flex items-center gap-4">
        <h1 className="text-lg font-semibold">Store Locations</h1>
        <select
          value={merchant}
          onChange={(e) => setMerchant(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
        >
          <option value="">All Merchants</option>
          <option value="arcaplanet">Arcaplanet</option>
          <option value="twinset">Twinset</option>
        </select>
        <span className="text-sm text-gray-500">{stores.length} stores</span>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 size={24} className="animate-spin text-gray-400" />
        </div>
      ) : stores.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-gray-400">
          <p>No stores found. Import store data with:<br />
            <code className="text-xs bg-gray-100 px-2 py-1 rounded mt-1 inline-block">
              python scripts/import_stores.py data/stores.xlsx
            </code>
          </p>
        </div>
      ) : (
        <div className="flex-1">
          <MapContainer
            center={center}
            zoom={6}
            className="h-full w-full"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {stores.map((store) => (
              <Marker
                key={store.store_id}
                position={[store.lat, store.lng]}
                icon={merchantIcon[store.merchant] || greenIcon}
              >
                <Popup>
                  <div className="text-sm">
                    <p className="font-semibold">{store.display_name || store.merchant}</p>
                    <p className="text-gray-600 capitalize">{store.merchant}</p>
                    {store.street && (
                      <p className="text-gray-500">
                        {store.street} {store.street_number}
                        {store.zip_code && `, ${store.zip_code}`}
                      </p>
                    )}
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      )}
    </div>
  )
}
