import React, { useState, useCallback } from 'react'
import { gameApi, GameStateResponse, ActionRequest, PlayerConfig } from './api/gameApi'
import { useGameSocket } from './hooks/useGameSocket'
import { Board } from './components/Board'
import { InfoPanel } from './components/InfoPanel'
import { ActionBar } from './components/ActionBar'

const HUMAN_PLAYER_ID = 0

type SetupState = {
  boardConfig: 'small_20' | 'classic_42' | 'grid_6x6'
  opponent: 'random' | 'rule_based' | 'mcts' | 'rl'
  numPlayers: number
}

function SetupScreen({ onStart }: { onStart: (s: SetupState) => void }) {
  const [setup, setSetup] = useState<SetupState>({
    boardConfig: 'grid_6x6',
    opponent: 'rule_based',
    numPlayers: 2,
  })

  const select = (style: React.CSSProperties): React.CSSProperties => ({
    padding: '6px 10px', borderRadius: 6, border: '1px solid #aaa',
    background: '#1a1a2e', color: '#fff', fontSize: 14, ...style,
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
                  justifyContent: 'center', height: '100vh', gap: 20, color: '#eee' }}>
      <h1 style={{ fontSize: 32 }}>Risk Game Bots</h1>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14,
                    background: '#0f3460', padding: 32, borderRadius: 12, minWidth: 320 }}>
        <label>
          <div style={{ marginBottom: 4, color: '#aaa', fontSize: 13 }}>Board</div>
          <select style={select({})} value={setup.boardConfig}
            onChange={e => setSetup(s => ({ ...s, boardConfig: e.target.value as any }))}>
            <option value="grid_6x6">Grid 6×6 (36 territories) — Classic</option>
            <option value="small_20">Small (20 territories) — Fast</option>
            <option value="classic_42">Classic Risk (42 territories)</option>
          </select>
        </label>
        <label>
          <div style={{ marginBottom: 4, color: '#aaa', fontSize: 13 }}>Opponent</div>
          <select style={select({})} value={setup.opponent}
            onChange={e => setSetup(s => ({ ...s, opponent: e.target.value as any }))}>
            <option value="random">Random Bot (easy)</option>
            <option value="rule_based">Rule-Based Bot (medium)</option>
            <option value="mcts">MCTS Bot (hard, slower)</option>
            <option value="rl">PyTorch AI (trained)</option>
          </select>
        </label>
        <label>
          <div style={{ marginBottom: 4, color: '#aaa', fontSize: 13 }}>Number of Players</div>
          <select style={select({})} value={setup.numPlayers}
            onChange={e => setSetup(s => ({ ...s, numPlayers: Number(e.target.value) }))}>
            {[2, 3, 4].map(n => <option key={n} value={n}>{n} Players</option>)}
          </select>
        </label>
        <button
          onClick={() => onStart(setup)}
          style={{ padding: '10px 20px', borderRadius: 8, border: 'none',
                   background: '#e74c3c', color: '#fff', fontSize: 16,
                   fontWeight: 'bold', cursor: 'pointer', marginTop: 8 }}
        >
          Start Game
        </button>
      </div>
    </div>
  )
}

export default function App() {
  const [gameId, setGameId] = useState<string | null>(null)
  const [manualState, setManualState] = useState<GameStateResponse | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [attackSrc, setAttackSrc] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { gameState: wsState, setGameState: setWsState, connected } = useGameSocket(gameId)
  const gameState = wsState ?? manualState

  const handleStart = useCallback(async (setup: SetupState) => {
    setError(null)
    setLoading(true)
    try {
      const players: PlayerConfig[] = [
        { type: 'human' },
        ...Array(setup.numPlayers - 1).fill(null).map(() => ({ type: setup.opponent })),
      ]
      const resp = await gameApi.createGame({ board_config: setup.boardConfig, players })
      setGameId(resp.game_id)
      // Fetch initial state
      const state = await gameApi.getGame(resp.game_id)
      setManualState(state)
    } catch (e: any) {
      setError('Failed to start game: ' + (e?.message ?? String(e)))
    }
    setLoading(false)
  }, [])

  const submitAction = useCallback(async (action: ActionRequest) => {
    if (!gameId) return
    setLoading(true)
    setError(null)
    try {
      const state = await gameApi.submitAction(gameId, action)
      setManualState(state)
      setWsState(state)
      setSelectedId(null)
      setAttackSrc(null)
    } catch (e: any) {
      setError('Action failed: ' + (e?.response?.data?.detail ?? e?.message ?? String(e)))
    }
    setLoading(false)
  }, [gameId])

  const handleEndPhase = useCallback(async () => {
    if (!gameState) return
    const endAction: ActionRequest = { phase: gameState.phase, troops: -1 }
    await submitAction(endAction)
  }, [gameState, submitAction])

  const handleTerritoryClick = useCallback((id: number) => {
    if (!gameState || gameState.status === 'finished') return
    if (gameState.current_player !== HUMAN_PLAYER_ID) return

    const territory = gameState.territories.find(t => t.id === id)
    if (!territory) return

    if (gameState.phase === 'DRAFT') {
      if (territory.owner === HUMAN_PLAYER_ID) setSelectedId(id)
    } else if (gameState.phase === 'ATTACK') {
      if (territory.owner === HUMAN_PLAYER_ID && territory.troops >= 2) {
        setAttackSrc(id)
        setSelectedId(id)
      } else if (attackSrc !== null && territory.owner !== HUMAN_PLAYER_ID) {
        // Check adjacency
        const src = gameState.territories.find(t => t.id === attackSrc)
        if (src && src.adjacent.includes(id)) {
          const dice = Math.min(src.troops - 1, 3)
          submitAction({ phase: 'ATTACK', src: attackSrc, dst: id, troops: dice })
        }
      }
    } else if (gameState.phase === 'FORTIFY') {
      if (!selectedId) {
        if (territory.owner === HUMAN_PLAYER_ID && territory.troops >= 2) setSelectedId(id)
      } else if (territory.id !== selectedId && territory.owner === HUMAN_PLAYER_ID) {
        submitAction({ phase: 'FORTIFY', src: selectedId, dst: id, troops: 1 })
        setSelectedId(null)
      }
    }
  }, [gameState, attackSrc, selectedId, submitAction])

  if (!gameId || !gameState) {
    return (
      <>
        {error && <div style={{ position: 'fixed', top: 16, right: 16, background: '#e74c3c',
                                 color: '#fff', padding: '10px 16px', borderRadius: 8 }}>{error}</div>}
        {loading && <div style={{ position: 'fixed', top: 16, right: 16, color: '#aaa' }}>Loading...</div>}
        <SetupScreen onStart={handleStart} />
      </>
    )
  }

  return (
    <div style={{ padding: 20, minHeight: '100vh' }}>
      {error && (
        <div style={{ background: '#e74c3c', color: '#fff', padding: '10px 16px',
                       borderRadius: 8, marginBottom: 12 }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: 12, background: 'none',
                                                           border: 'none', color: '#fff', cursor: 'pointer' }}>✕</button>
        </div>
      )}
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Board
            territories={gameState.territories}
            grid={gameState.grid}
            selectedId={selectedId}
            attackSrc={attackSrc}
            onTerritoryClick={handleTerritoryClick}
            currentPlayer={gameState.current_player}
            phase={gameState.phase}
          />
          <ActionBar
            gameState={gameState}
            selectedId={selectedId}
            humanPlayerId={HUMAN_PLAYER_ID}
            onAction={submitAction}
            onEndPhase={handleEndPhase}
            loading={loading}
          />
        </div>
        <InfoPanel gameState={gameState} />
      </div>
      {!connected && gameId && (
        <div style={{ position: 'fixed', bottom: 16, right: 16,
                       background: '#555', color: '#eee', padding: '6px 12px', borderRadius: 6, fontSize: 12 }}>
          WebSocket disconnected
        </div>
      )}
    </div>
  )
}
