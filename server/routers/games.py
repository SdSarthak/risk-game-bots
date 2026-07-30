"""REST endpoints for game management."""
from fastapi import APIRouter, HTTPException, Request

from server.schemas import ActionRequest, GameCreateRequest, GameCreateResponse, GameStateResponse

router = APIRouter(prefix="/games", tags=["games"])


def get_manager(request: Request):
    return request.app.state.game_manager


@router.post("", response_model=GameCreateResponse)
def create_game(body: GameCreateRequest, request: Request):
    manager = get_manager(request)
    if len(body.players) < 2 or len(body.players) > 6:
        raise HTTPException(status_code=400, detail="Must have 2–6 players")
    game_id, _ = manager.create_game(body)
    return manager.build_create_response(game_id)


@router.get("/{game_id}", response_model=GameStateResponse)
def get_game(game_id: str, request: Request):
    manager = get_manager(request)
    session = manager.get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return manager._build_response(session)


@router.post("/{game_id}/action", response_model=GameStateResponse)
def submit_action(game_id: str, body: ActionRequest, request: Request):
    manager = get_manager(request)
    session = manager.get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if manager._build_response(session).status == "finished":
        raise HTTPException(status_code=400, detail="Game is already finished")
    try:
        state_response = manager.apply_human_action(game_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state_response


@router.post("/{game_id}/step", response_model=GameStateResponse)
def step_bots(game_id: str, request: Request):
    """Advance the game by one full bot turn sequence (for all-bot games)."""
    manager = get_manager(request)
    session = manager.get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return manager.step_bots(game_id)
