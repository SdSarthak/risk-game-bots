// Single source of player colours. The board and the info panel used to keep
// their own copies, so the same player was drawn in two different reds.
export const PLAYER_COLORS = [
  '#e74c3c', // P0 Red
  '#3498db', // P1 Blue
  '#2ecc71', // P2 Green
  '#f39c12', // P3 Orange
  '#9b59b6', // P4 Purple
  '#1abc9c', // P5 Teal
]

export const NEUTRAL_COLOR = '#555'

export function playerColor(owner: number): string {
  if (owner < 0) return NEUTRAL_COLOR
  return PLAYER_COLORS[owner % PLAYER_COLORS.length]
}
