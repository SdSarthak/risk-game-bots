import React from 'react'
import type { GameStateResponse, ActionRequest } from '../api/gameApi'

interface ActionBarProps {
  gameState: GameStateResponse
  selectedId: number | null
  humanPlayerId: number
  /** Troop/dice count the next action will use. Owned by App so clicks on the board can read it. */
  troops: number
  onTroopsChange: (troops: number) => void
  onAction: (action: ActionRequest) => void
  onEndPhase: () => void
  loading: boolean
}

export function ActionBar({ gameState, selectedId, humanPlayerId, troops,
                             onTroopsChange, onAction, onEndPhase, loading }: ActionBarProps) {
  const { phase, current_player, troops_to_place, status } = gameState
  const selected = selectedId === null
    ? undefined
    : gameState.territories.find(t => t.id === selectedId)

  if (status === 'finished') return null
  if (current_player !== humanPlayerId) {
    return (
      <div style={barStyle}>
        <span style={{ color: '#aaa' }}>Waiting for bots...</span>
      </div>
    )
  }

  const btn = (label: string, onClick: () => void, disabled = false, accent = false) => (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        padding: '8px 18px',
        borderRadius: 6,
        border: 'none',
        background: disabled ? '#333' : accent ? '#e74c3c' : '#0f3460',
        color: disabled ? '#666' : '#fff',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontWeight: 'bold',
        fontSize: 14,
      }}
    >
      {label}
    </button>
  )

  const numberInput = (max: number) => (
    <input
      type="number" min={1} max={max} value={Math.min(troops, max)}
      onChange={e => onTroopsChange(clamp(Number(e.target.value), 1, max))}
      style={inputStyle}
    />
  )

  if (phase === 'DRAFT') {
    return (
      <div style={barStyle}>
        <span style={{ color: '#ffd700' }}>Place up to {troops_to_place} troops</span>
        {numberInput(Math.max(1, troops_to_place))}
        {btn('Place on Selected', () => {
          if (selectedId === null) return
          onAction({ phase: 'DRAFT', dst: selectedId, troops: clamp(troops, 1, troops_to_place) })
        }, selectedId === null || troops_to_place < 1)}
        {btn('End Draft', onEndPhase, troops_to_place > 0)}
      </div>
    )
  }

  if (phase === 'ATTACK') {
    // A garrison must leave one army behind, and Risk caps an attack at three dice
    const maxDice = selected ? clamp(selected.troops - 1, 1, 3) : 3
    return (
      <div style={barStyle}>
        <span style={{ color: '#aaa' }}>
          Select your territory, then click an adjacent enemy to attack
        </span>
        {numberInput(maxDice)}
        <span style={{ color: '#888', fontSize: 13 }}>
          dice{selected ? ` (max ${maxDice} from ${selected.name})` : ''}
        </span>
        {btn('End Attack', onEndPhase)}
      </div>
    )
  }

  if (phase === 'FORTIFY') {
    const maxMove = selected ? Math.max(1, selected.troops - 1) : 1
    return (
      <div style={barStyle}>
        <span style={{ color: '#aaa' }}>
          Select a source, then a connected territory of yours to reinforce
        </span>
        {numberInput(maxMove)}
        <span style={{ color: '#888', fontSize: 13 }}>
          troops{selected ? ` (max ${maxMove} from ${selected.name})` : ''}
        </span>
        {btn('End Turn', onEndPhase)}
      </div>
    )
  }

  return null
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min
  return Math.min(max, Math.max(min, value))
}

const barStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  background: '#0f3460',
  borderRadius: 10,
  padding: '12px 20px',
  flexWrap: 'wrap',
  color: '#eee',
}

const inputStyle: React.CSSProperties = {
  width: 64,
  padding: '6px 8px',
  borderRadius: 6,
  border: '1px solid #aaa',
  background: '#1a1a2e',
  color: '#fff',
  fontSize: 14,
}
