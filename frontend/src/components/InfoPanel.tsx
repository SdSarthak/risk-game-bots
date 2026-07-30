import React from 'react'
import type { GameStateResponse } from '../api/gameApi'
import { PLAYER_COLORS } from './Territory'

interface InfoPanelProps {
  gameState: GameStateResponse
}

export function InfoPanel({ gameState }: InfoPanelProps) {
  const { current_player, phase, troops_to_place, players, status, winner, turn_number } = gameState

  const phaseLabel: Record<string, string> = {
    DRAFT: 'Draft — Place Troops',
    ATTACK: 'Attack',
    FORTIFY: 'Fortify',
  }

  return (
    <div style={{
      background: '#0f3460',
      borderRadius: 10,
      padding: '16px 20px',
      minWidth: 220,
      color: '#eee',
    }}>
      <h2 style={{ fontSize: 18, marginBottom: 12 }}>Risk</h2>

      {status === 'finished' ? (
        <div style={{ color: PLAYER_COLORS[winner ?? 0], fontWeight: 'bold', fontSize: 16 }}>
          Player {winner} wins!
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 8 }}>
            <span style={{ color: '#aaa', fontSize: 12 }}>Turn</span>
            <span style={{ marginLeft: 8, fontWeight: 'bold' }}>{turn_number}</span>
          </div>
          <div style={{
            background: PLAYER_COLORS[current_player % 6] + '33',
            border: `2px solid ${PLAYER_COLORS[current_player % 6]}`,
            borderRadius: 8,
            padding: '8px 12px',
            marginBottom: 12,
          }}>
            <div style={{ fontSize: 12, color: '#aaa' }}>Current Player</div>
            <div style={{ color: PLAYER_COLORS[current_player % 6], fontWeight: 'bold' }}>
              Player {current_player} ({players[current_player]?.type})
            </div>
            <div style={{ fontSize: 13, marginTop: 4 }}>{phaseLabel[phase] ?? phase}</div>
            {phase === 'DRAFT' && troops_to_place > 0 && (
              <div style={{ color: '#ffd700', fontSize: 13, marginTop: 4 }}>
                Troops to place: {troops_to_place}
              </div>
            )}
          </div>
        </>
      )}

      <div>
        <div style={{ fontSize: 12, color: '#aaa', marginBottom: 6 }}>Players</div>
        {players.map(p => (
          <div key={p.id} style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            marginBottom: 6,
            opacity: p.eliminated ? 0.4 : 1,
          }}>
            <div style={{
              width: 12, height: 12, borderRadius: '50%',
              background: PLAYER_COLORS[p.id % 6],
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 13, flex: 1 }}>
              P{p.id} {p.is_human ? '(You)' : `(${p.type})`}
            </span>
            <span style={{ fontSize: 12, color: '#aaa' }}>
              {p.territory_count}T / {p.troop_count}A
            </span>
            {p.eliminated && <span style={{ fontSize: 10, color: '#e74c3c' }}>✗</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
