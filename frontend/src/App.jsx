import { Routes, Route, Navigate } from 'react-router-dom'
import { SessionProvider, useSession } from './context/SessionContext'
import Navbar from './components/Navbar'
import LandingPage from './pages/LandingPage'
import RecommenderPage from './pages/RecommenderPage'
import BrowsePage from './pages/BrowsePage'
import MapPage from './pages/MapPage'
import EvalPage from './pages/EvalPage'

function ProtectedRoutes() {
  const { hasSelectedProfile } = useSession()

  if (!hasSelectedProfile) {
    return <Navigate to="/" replace />
  }

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/recommend" element={<RecommenderPage />} />
          <Route path="/browse" element={<BrowsePage />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/eval" element={<EvalPage />} />
          <Route path="*" element={<Navigate to="/recommend" replace />} />
        </Routes>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <SessionProvider>
      <Routes>
        <Route path="/" element={<LandingPageWrapper />} />
        <Route path="/*" element={<ProtectedRoutes />} />
      </Routes>
    </SessionProvider>
  )
}

function LandingPageWrapper() {
  const { hasSelectedProfile } = useSession()

  if (hasSelectedProfile) {
    return <Navigate to="/recommend" replace />
  }

  return <LandingPage />
}
