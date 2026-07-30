import React, { useMemo } from 'react'
import type { TerritoryState, GridInfo } from '../api/gameApi'

// Color palette for players (up to 6)
const PLAYER_COLORS = [
  '#c0392b', // P0 Red
  '#2980b9', // P1 Blue
  '#27ae60', // P2 Green
  '#d35400', // P3 Orange
  '#8e44ad', // P4 Purple
  '#16a085', // P5 Teal
]

const NEUTRAL_COLOR = '#444'
const CONTINENT_COLORS: Record<string, string> = {
  NW: '#1a3a2a', NE: '#1a2a3a', SW: '#2a1a3a', SE: '#3a2a1a',
  West: '#1a3a2a', East: '#1a2a3a', North: '#2a1a3a', South: '#3a2a1a',
}

interface BoardProps {
  territories: TerritoryState[]
  grid: GridInfo | null
  selectedId: number | null
  attackSrc: number | null
  onTerritoryClick: (id: number) => void
  currentPlayer: number
  phase: string
}

// ── Grid board layout ──────────────────────────────────────────────────────────

const CELL = 90       // cell size in px
const PADDING = 24    // outer padding
const RADIUS = 32     // territory circle radius

function GridBoard({ territories, grid, selectedId, attackSrc, onTerritoryClick,
                     currentPlayer, phase }: BoardProps & { grid: GridInfo }) {
  const W = grid.cols * CELL + PADDING * 2
  const H = grid.rows * CELL + PADDING * 2

  // Map id → territory
  const byId = useMemo(() => {
    const m: Record<number, TerritoryState> = {}
    territories.forEach(t => { m[t.id] = t })
    return m
  }, [territories])

  // Which territories are valid targets for the current selection
  const validTargets = useMemo(() => {
    if (phase !== 'ATTACK' || attackSrc === null) return new Set<number>()
    const src = byId[attackSrc]
    if (!src || src.owner !== currentPlayer) return new Set<number>()
    return new Set(src.adjacent.filter(id => byId[id]?.owner !== currentPlayer))
  }, [phase, attackSrc, byId, currentPlayer])

  const fortifyTargets = useMemo(() => {
    if (phase !== 'FORTIFY' || selectedId === null) return new Set<number>()
    const src = byId[selectedId]
    if (!src || src.owner !== currentPlayer) return new Set<number>()
    return new Set(src.adjacent.filter(id => byId[id]?.owner === currentPlayer && id !== selectedId))
  }, [phase, selectedId, byId, currentPlayer])

  const cx = (col: number) => PADDING + col * CELL + CELL / 2
  const cy = (row: number) => PADDING + row * CELL + CELL / 2

  return (
    <div style={{
      background: '#0d1117',
      borderRadius: 12,
      overflow: 'hidden',
      boxShadow: '0 4px 32px rgba(0,0,0,0.7)',
      display: 'inline-block',
    }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {/* Continent background tinting */}
        {territories.map(t => {
          if (t.row == null || t.col == null) return null
          const bg = CONTINENT_COLORS[t.continent] ?? '#1a1a2e'
          return (
            <rect
              key={`bg-${t.id}`}
              x={PADDING + t.col * CELL}
              y={PADDING + t.row * CELL}
              width={CELL}
              height={CELL}
              fill={bg}
              opacity={0.6}
            />
          )
        })}

        {/* Grid lines */}
        {Array.from({ length: grid.rows + 1 }, (_, r) => (
          <line key={`hr${r}`}
            x1={PADDING} y1={PADDING + r * CELL}
            x2={PADDING + grid.cols * CELL} y2={PADDING + r * CELL}
            stroke="rgba(255,255,255,0.08)" strokeWidth={1}
          />
        ))}
        {Array.from({ length: grid.cols + 1 }, (_, c) => (
          <line key={`vc${c}`}
            x1={PADDING + c * CELL} y1={PADDING}
            x2={PADDING + c * CELL} y2={PADDING + grid.rows * CELL}
            stroke="rgba(255,255,255,0.08)" strokeWidth={1}
          />
        ))}

        {/* Adjacency connectors (horizontal & vertical lines between cells) */}
        {territories.map(t => {
          if (t.row == null || t.col == null) return null
          return t.adjacent.map(adjId => {
            if (adjId <= t.id) return null
            const adj = byId[adjId]
            if (adj?.row == null || adj?.col == null) return null
            return (
              <line
                key={`edge-${t.id}-${adjId}`}
                x1={cx(t.col!)} y1={cy(t.row!)}
                x2={cx(adj.col!)} y2={cy(adj.row!)}
                stroke="rgba(255,255,255,0.18)"
                strokeWidth={2}
              />
            )
          })
        })}

        {/* Territory cells */}
        {territories.map(t => {
          if (t.row == null || t.col == null) return null
          const x = cx(t.col)
          const y = cy(t.row)
          const isSelected = t.id === selectedId || t.id === attackSrc
          const isTarget = validTargets.has(t.id) || fortifyTargets.has(t.id)
          const playerColor = t.owner >= 0 ? PLAYER_COLORS[t.owner % 6] : NEUTRAL_COLOR
          const strokeColor = isSelected ? '#ffffff' : isTarget ? '#ffd700' : 'rgba(255,255,255,0.2)'
          const strokeWidth = isSelected || isTarget ? 3 : 1.5
          const pulseRing = isTarget

          return (
            <g
              key={t.id}
              onClick={() => onTerritoryClick(t.id)}
              style={{ cursor: 'pointer' }}
            >
              {/* Pulse ring for valid targets */}
              {pulseRing && (
                <circle
                  cx={x} cy={y} r={RADIUS + 6}
                  fill="none"
                  stroke="#ffd700"
                  strokeWidth={2}
                  opacity={0.5}
                />
              )}

              {/* Main territory circle */}
              <circle
                cx={x} cy={y} r={RADIUS}
                fill={playerColor}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                opacity={t.owner < 0 ? 0.35 : 1}
              />

              {/* Selection glow */}
              {isSelected && (
                <circle cx={x} cy={y} r={RADIUS} fill="rgba(255,255,255,0.15)" />
              )}

              {/* Troop count */}
              <text
                x={x} y={y}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#fff"
                fontSize={18}
                fontWeight="bold"
                fontFamily="monospace"
                style={{ userSelect: 'none', pointerEvents: 'none' }}
              >
                {t.troops}
              </text>

              {/* Continent label (tiny, bottom-right corner of cell) */}
              <text
                x={PADDING + t.col * CELL + CELL - 4}
                y={PADDING + t.row * CELL + 12}
                textAnchor="end"
                fill="rgba(255,255,255,0.3)"
                fontSize={8}
                style={{ userSelect: 'none', pointerEvents: 'none' }}
              >
                {t.continent}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ── Auto-layout fallback for non-grid boards ───────────────────────────────────

function computeAutoLayout(territories: TerritoryState[]): Record<number, { x: number; y: number }> {
  const n = territories.length
  const cols = Math.ceil(Math.sqrt(n * 1.6))
  const W = 800, H = 600
  const xStep = W / (cols + 1)
  const yStep = H / (Math.ceil(n / cols) + 1)
  const layout: Record<number, { x: number; y: number }> = {}
  territories.forEach((t, i) => {
    const col = (i % cols) + 1
    const row = Math.floor(i / cols) + 1
    const jx = ((t.id * 37) % 20) - 10
    const jy = ((t.id * 53) % 20) - 10
    layout[t.id] = { x: col * xStep + jx, y: row * yStep + jy }
  })
  return layout
}

function AutoLayoutBoard({ territories, selectedId, attackSrc, onTerritoryClick,
                            currentPlayer, phase }: Omit<BoardProps, 'grid'>) {
  const layout = useMemo(() => computeAutoLayout(territories), [territories.length])
  const byId = useMemo(() => {
    const m: Record<number, TerritoryState> = {}
    territories.forEach(t => { m[t.id] = t })
    return m
  }, [territories])

  const validTargets = useMemo(() => {
    if (phase !== 'ATTACK' || attackSrc === null) return new Set<number>()
    const src = byId[attackSrc]
    if (!src || src.owner !== currentPlayer) return new Set<number>()
    return new Set(src.adjacent.filter(id => byId[id]?.owner !== currentPlayer))
  }, [phase, attackSrc, byId, currentPlayer])

  const W = 800, H = 600

  return (
    <div style={{ background: '#16213e', borderRadius: 12, overflow: 'hidden', boxShadow: '0 4px 24px rgba(0,0,0,0.5)' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {territories.map(t =>
          t.adjacent.map(adjId => {
            if (adjId <= t.id) return null
            const from = layout[t.id], to = layout[adjId]
            if (!from || !to) return null
            return (
              <line key={`${t.id}-${adjId}`}
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke="rgba(255,255,255,0.12)" strokeWidth={1.5}
              />
            )
          })
        )}
        {territories.map(t => {
          const pos = layout[t.id]
          if (!pos) return null
          const isSelected = t.id === selectedId || t.id === attackSrc
          const isTarget = validTargets.has(t.id)
          const color = t.owner >= 0 ? PLAYER_COLORS[t.owner % 6] : NEUTRAL_COLOR
          return (
            <g key={t.id} onClick={() => onTerritoryClick(t.id)} style={{ cursor: 'pointer' }}>
              <circle cx={pos.x} cy={pos.y} r={28}
                fill={color}
                stroke={isSelected ? '#fff' : isTarget ? '#ffd700' : 'rgba(255,255,255,0.3)'}
                strokeWidth={isSelected || isTarget ? 3 : 1.5}
                opacity={t.owner < 0 ? 0.4 : 0.9}
              />
              <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="central"
                fill="#fff" fontSize={14} fontWeight="bold"
                style={{ userSelect: 'none', pointerEvents: 'none' }}>
                {t.troops}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ── Main Board component ───────────────────────────────────────────────────────

export function Board(props: BoardProps) {
  if (props.grid) {
    return <GridBoard {...props} grid={props.grid} />
  }
  return <AutoLayoutBoard {...props} />
}

export { PLAYER_COLORS }
