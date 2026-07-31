import axios from 'axios'

// Point the UI at a different backend with VITE_API_BASE (see .env.example).
export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

const BASE = API_BASE.replace(/\/+$/, '')

export interface PlayerConfig {
  type: 'human' | 'random' | 'rule_based' | 'mcts' | 'rl'
  checkpoint?: string
}

export interface GameCreateRequest {
  board_config: 'small_20' | 'classic_42' | 'grid_6x6'
  players: PlayerConfig[]
  seed?: number
}

export interface TerritoryState {
  id: number
  name: string
  continent: string
  owner: number
  troops: number
  adjacent: number[]
  row: number | null
  col: number | null
}

export interface GridInfo {
  rows: number
  cols: number
}

export interface PlayerState {
  id: number
  type: string
  is_human: boolean
  eliminated: boolean
  territory_count: number
  troop_count: number
  card_count: number
}

export interface GameStateResponse {
  game_id: string
  status: 'active' | 'finished'
  winner: number | null
  current_player: number
  phase: 'DRAFT' | 'ATTACK' | 'FORTIFY'
  troops_to_place: number
  territories: TerritoryState[]
  players: PlayerState[]
  turn_number: number
  grid: GridInfo | null
}

export interface GameCreateResponse {
  game_id: string
  board_name: string
  num_territories: number
  players: PlayerState[]
}

export interface ActionRequest {
  phase: 'DRAFT' | 'ATTACK' | 'FORTIFY'
  src?: number
  dst?: number
  troops: number
}

export interface ActionOption {
  phase: 'DRAFT' | 'ATTACK' | 'FORTIFY'
  src: number
  dst: number
  troops: number
  end_phase: boolean
}

export interface LegalActionsResponse {
  game_id: string
  current_player: number
  phase: 'DRAFT' | 'ATTACK' | 'FORTIFY'
  actions: ActionOption[]
  /** The server caps the list; true when more legal moves exist than were sent. */
  truncated: boolean
  limit: number
}

export const gameApi = {
  createGame: (body: GameCreateRequest) =>
    axios.post<GameCreateResponse>(`${BASE}/games`, body).then(r => r.data),

  getGame: (gameId: string) =>
    axios.get<GameStateResponse>(`${BASE}/games/${gameId}`).then(r => r.data),

  getLegalActions: (gameId: string, limit?: number) =>
    axios
      .get<LegalActionsResponse>(`${BASE}/games/${gameId}/legal-actions`,
        limit === undefined ? undefined : { params: { limit } })
      .then(r => r.data),

  submitAction: (gameId: string, action: ActionRequest) =>
    axios.post<GameStateResponse>(`${BASE}/games/${gameId}/action`, action).then(r => r.data),

  stepBots: (gameId: string) =>
    axios.post<GameStateResponse>(`${BASE}/games/${gameId}/step`).then(r => r.data),
}
