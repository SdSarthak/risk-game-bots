import React from 'react'
import type { TerritoryState } from '../api/gameApi'

// Color palette for players (up to 6)
const PLAYER_COLORS = [
  '#e74c3c', // P0 Red
  '#3498db', // P1 Blue
  '#2ecc71', // P2 Green
  '#f39c12', // P3 Orange
  '#9b59b6', // P4 Purple
  '#1abc9c', // P5 Teal
]

const NEUTRAL_COLOR = '#555'

interface TerritoryProps {
  territory: TerritoryState
  x: number
  y: number
  radius?: number
  isSelected: boolean
  isHighlighted: boolean  // valid attack/fortify target
  onClick: (id: number) => void
}

export function Territory({ territory, x, y, radius = 28, isSelected,
                             isHighlighted, onClick }: TerritoryProps) {
  const color = territory.owner >= 0 ? PLAYER_COLORS[territory.owner % 6] : NEUTRAL_COLOR
  const borderColor = isSelected ? '#fff' : isHighlighted ? '#ffd700' : 'rgba(255,255,255,0.3)'
  const borderWidth = isSelected || isHighlighted ? 3 : 1.5

  return (
    <g
      onClick={() => onClick(territory.id)}
      style={{ cursor: 'pointer' }}
      role="button"
      aria-label={`${territory.name}: ${territory.troops} troops`}
    >
      <circle
        cx={x} cy={y} r={radius}
        fill={color}
        stroke={borderColor}
        strokeWidth={borderWidth}
        opacity={territory.owner < 0 ? 0.4 : 0.9}
      />
      {/* Troop count */}
      <text
        x={x} y={y - 4}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#fff"
        fontSize={14}
        fontWeight="bold"
      >
        {territory.troops}
      </text>
      {/* Territory name (abbreviated) */}
      <text
        x={x} y={y + 10}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="rgba(255,255,255,0.8)"
        fontSize={8}
      >
        {territory.name.slice(0, 3).toUpperCase()}
      </text>
    </g>
  )
}

export { PLAYER_COLORS }
