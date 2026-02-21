import { useState, useRef, useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Sparkles, ShoppingBag, Map, BarChart3, LogOut, User } from 'lucide-react'
import { useSession } from '../context/SessionContext'
import ProfilePicker from './ProfilePicker'

const links = [
  { to: '/recommend', label: 'Recommend', icon: Sparkles },
  { to: '/browse', label: 'Browse', icon: ShoppingBag },
  { to: '/map', label: 'Map', icon: Map },
  { to: '/eval', label: 'Evaluation', icon: BarChart3 },
]

export default function Navbar() {
  const { profile, selectProfile, logout } = useSession()
  const navigate = useNavigate()
  const [showDropdown, setShowDropdown] = useState(false)
  const dropdownRef = useRef(null)

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false)
      }
    }
    if (showDropdown) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showDropdown])

  const handleSignOut = () => {
    setShowDropdown(false)
    logout()
    navigate('/')
  }

  const handleProfileSwitch = (p) => {
    selectProfile(p)
    setShowDropdown(false)
  }

  return (
    <nav className="border-b border-gray-200 bg-white px-6 py-3">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        <span className="text-lg font-semibold text-gray-900">
          Product Recommender
        </span>
        <div className="flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}

          {/* Profile badge + dropdown */}
          <div className="relative ml-3" ref={dropdownRef}>
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-full bg-gray-100 text-sm text-gray-700 hover:bg-gray-200 transition-colors"
            >
              {profile ? (
                <>
                  <span>{profile.emoji}</span>
                  <span className="font-medium">{profile.name}</span>
                </>
              ) : (
                <>
                  <User size={14} />
                  <span className="font-medium">Guest</span>
                </>
              )}
            </button>

            {showDropdown && (
              <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-xl shadow-lg border border-gray-200 p-4 z-50">
                <p className="text-xs text-gray-500 mb-3 font-medium">Switch profile</p>
                <ProfilePicker
                  selectedId={profile?.profile_id}
                  onSelect={handleProfileSwitch}
                />
                <hr className="my-3 border-gray-200" />
                <button
                  onClick={handleSignOut}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-md transition-colors"
                >
                  <LogOut size={14} />
                  Sign Out
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
