import React, { useState } from 'react'
import type { GameStateResponse, ActionRequest } from '../api/gameApi'

interface ActionBarProps {
  gameState: GameStateResponse
  selectedId: number | null
  humanPlayerId: number
  onAction: (action: ActionRequest) => void
  onEndPhase: () => void
  loading: boolean
}

export function ActionBar({ gameState, selectedId, humanPlayerId,
                             onAction, onEndPhase, loading }: ActionBarProps) {
  const [troops, setTroops] = useState(1)
  const { phase, current_player, troops_to_place, status } = gameState

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

  if (phase === 'DRAFT') {
    return (
      <div style={barStyle}>
        <span style={{ color: '#ffd700' }}>Place up to {troops_to_place} troops</span>
        <input
          type="number" min={1} max={troops_to_place} value={troops}
          onChange={e => setTroops(Number(e.target.value))}
          style={inputStyle}
        />
        {btn('Place on Selected', () => {
          if (selectedId === null) return
          onAction({ phase: 'DRAFT', dst: selectedId, troops: Math.min(troops, troops_to_place) })
        }, selectedId === null)}
        {btn('End Draft', onEndPhase)}
      </div>
    )
  }

  if (phase === 'ATTACK') {
    return (
      <div style={barStyle}>
        <span style={{ color: '#aaa' }}>Select your territory, then click an enemy to attack</span>
        <input
          type="number" min={1} max={3} value={troops}
          onChange={e => setTroops(Math.min(3, Math.max(1, Number(e.target.value))))}
          style={inputStyle}
        />
        {btn('Attack (dice: ' + troops + ')', () => {
          // Action is submitted from App when target is clicked
        }, true)}
        {btn('End Attack', onEndPhase)}
      </div>
    )
  }

  if (phase === 'FORTIFY') {
    return (
      <div style={barStyle}>
        <span style={{ color: '#aaa' }}>Select source, then destination to fortify</span>
        <input
          type="number" min={1} value={troops}
          onChange={e => setTroops(Math.max(1, Number(e.target.value)))}
          style={inputStyle}
        />
        {btn('End Turn', onEndPhase)}
      </div>
    )
  }

  return null
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
