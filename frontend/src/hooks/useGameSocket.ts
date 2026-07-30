import { useEffect, useRef, useState } from 'react'
import type { GameStateResponse } from '../api/gameApi'

import { API_BASE } from '../api/gameApi'

// Derive the socket origin from the API base so both follow VITE_API_BASE.
const WS_BASE = API_BASE.replace(/^http/, 'ws').replace(/\/+$/, '')

export function useGameSocket(gameId: string | null) {
  const [gameState, setGameState] = useState<GameStateResponse | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!gameId) return

    const ws = new WebSocket(`${WS_BASE}/ws/${gameId}`)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)

    ws.onmessage = (event) => {
      try {
        const state: GameStateResponse = JSON.parse(event.data)
        setGameState(state)
      } catch {
        console.error('Failed to parse WS message', event.data)
      }
    }

    ws.onclose = () => setConnected(false)
    ws.onerror = (e) => console.error('WebSocket error', e)

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [gameId])

  const ping = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send('ping')
    }
  }

  return { gameState, setGameState, connected, ping }
}
