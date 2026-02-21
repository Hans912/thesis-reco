import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const greenIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
})

const violetIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-violet.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
})

const merchantIcon = { arcaplanet: greenIcon, twinset: violetIcon }

export default function StoreMap({ stores }) {
  if (!stores || stores.length === 0) return null

  const avgLat = stores.reduce((s, st) => s + st.lat, 0) / stores.length
  const avgLng = stores.reduce((s, st) => s + st.lng, 0) / stores.length

  return (
    <div className="rounded-lg overflow-hidden border border-gray-200 mt-2" style={{ height: 250 }}>
      <MapContainer center={[avgLat, avgLng]} zoom={7} className="h-full w-full" scrollWheelZoom={false}>
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
                {store.address && <p className="text-gray-500">{store.address}</p>}
                {store.distance_km != null && (
                  <p className="text-gray-400">{store.distance_km} km away</p>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  )
}
