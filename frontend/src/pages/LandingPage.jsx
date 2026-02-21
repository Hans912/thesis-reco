import { useNavigate } from 'react-router-dom'
import { useSession } from '../context/SessionContext'
import ProfilePicker from '../components/ProfilePicker'

export default function LandingPage() {
  const { profile, selectProfile } = useSession()
  const navigate = useNavigate()

  const handleSelect = (p) => {
    selectProfile(p)
    navigate('/recommend')
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-blue-50 px-4">
      <div className="max-w-lg w-full text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Product Recommender</h1>
        <p className="text-gray-500 mb-8">
          Select a profile to get personalized store and product recommendations, or continue as Guest.
        </p>
        <ProfilePicker selectedId={profile?.profile_id} onSelect={handleSelect} />
      </div>
    </div>
  )
}
