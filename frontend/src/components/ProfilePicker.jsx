import { useState, useEffect } from 'react'
import { User } from 'lucide-react'

export default function ProfilePicker({ selectedId, onSelect }) {
  const [profiles, setProfiles] = useState([])

  useEffect(() => {
    fetch('/api/profiles')
      .then(r => r.json())
      .then(setProfiles)
      .catch(console.error)
  }, [])

  return (
    <div className="flex flex-wrap gap-3">
      {profiles.map((p) => (
        <button
          key={p.profile_id}
          onClick={() => onSelect(p)}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
            selectedId === p.profile_id
              ? 'border-blue-500 bg-blue-50 text-blue-700 ring-2 ring-blue-200'
              : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50'
          }`}
        >
          <span className="text-lg">{p.emoji}</span>
          <div className="text-left">
            <div className="font-semibold">{p.name}</div>
            <div className="text-xs text-gray-500 max-w-48 truncate">{p.description}</div>
          </div>
        </button>
      ))}
      <button
        onClick={() => onSelect(null)}
        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
          !selectedId
            ? 'border-blue-500 bg-blue-50 text-blue-700 ring-2 ring-blue-200'
            : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50'
        }`}
      >
        <User size={18} />
        <div className="text-left">
          <div className="font-semibold">Guest</div>
          <div className="text-xs text-gray-500">No purchase history</div>
        </div>
      </button>
    </div>
  )
}
