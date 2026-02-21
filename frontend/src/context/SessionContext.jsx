import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const SessionContext = createContext(null)

function getOrCreateSessionId() {
  let id = localStorage.getItem('session_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('session_id', id)
  }
  return id
}

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId)
  const [favorites, setFavorites] = useState(new Set())
  const [profile, setProfile] = useState(() => {
    const saved = localStorage.getItem('profile')
    return saved ? JSON.parse(saved) : null
  })
  const [hasSelectedProfile, setHasSelectedProfile] = useState(() => {
    return localStorage.getItem('hasSelectedProfile') === 'true'
  })

  const selectProfile = useCallback((p) => {
    setProfile(p)
    setHasSelectedProfile(true)
    localStorage.setItem('hasSelectedProfile', 'true')
    if (p) {
      localStorage.setItem('profile', JSON.stringify(p))
      setSessionId(p.profile_id)
      localStorage.setItem('session_id', p.profile_id)
    } else {
      localStorage.removeItem('profile')
      const guestId = crypto.randomUUID()
      setSessionId(guestId)
      localStorage.setItem('session_id', guestId)
    }
  }, [])

  const logout = useCallback(() => {
    setProfile(null)
    setHasSelectedProfile(false)
    localStorage.removeItem('profile')
    localStorage.removeItem('hasSelectedProfile')
    const guestId = crypto.randomUUID()
    setSessionId(guestId)
    localStorage.setItem('session_id', guestId)
  }, [])

  // Load favorites from API on mount / session change
  useEffect(() => {
    fetch(`/api/favorites?session_id=${sessionId}`)
      .then(r => r.ok ? r.json() : { favorites: [] })
      .then(data => setFavorites(new Set(data.favorites.map(f => f.product_id))))
      .catch(() => {})
  }, [sessionId])

  const toggleFavorite = useCallback(async (productId) => {
    const isFav = favorites.has(productId)
    setFavorites(prev => {
      const next = new Set(prev)
      if (isFav) next.delete(productId)
      else next.add(productId)
      return next
    })

    try {
      if (isFav) {
        await fetch(`/api/favorites/${productId}?session_id=${sessionId}`, { method: 'DELETE' })
      } else {
        await fetch('/api/favorites', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, product_id: productId }),
        })
      }
    } catch {
      setFavorites(prev => {
        const next = new Set(prev)
        if (isFav) next.add(productId)
        else next.delete(productId)
        return next
      })
    }
  }, [favorites, sessionId])

  const isFavorite = useCallback((productId) => favorites.has(productId), [favorites])

  return (
    <SessionContext.Provider value={{
      sessionId, setSessionId, favorites, toggleFavorite, isFavorite,
      profile, selectProfile, hasSelectedProfile, logout,
    }}>
      {children}
    </SessionContext.Provider>
  )
}

export function useSession() {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used within SessionProvider')
  return ctx
}
