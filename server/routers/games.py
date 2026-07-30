"""REST endpoints for game management."""
from fastapi import APIRouter, HTTPException, Request

from server.game_manager import GameManager
from server.schemas import (
    ActionRequest, GameCreateRequest, GameCreateResponse,
    GameStateResponse, LegalActionsResponse,
)

router = APIRouter(prefix="/games", tags=["games"])


def get_manager(request: Request) -> GameManager:
    return request.app.state.game_manager


def get_session(request: Request, game_id: str):
    manager = get_manager(request)
    session = manager.get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return manager, session


@router.post("", response_model=GameCreateResponse, status_code=201)
def create_game(body: GameCreateRequest, request: Request):
    manager = get_manager(request)
    try:
        game_id, _ = manager.create_game(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return manager.build_create_response(game_id)


@router.get("/{game_id}", response_model=GameStateResponse)
def get_game(game_id: str, request: Request):
    manager, session = get_session(request, game_id)
    return manager.build_state_response(session)


@router.get("/{game_id}/legal-actions", response_model=LegalActionsResponse)
def get_legal_actions(game_id: str, request: Request):
    """Every action the player to move may take — drives the UI's move picker."""
    manager, session = get_session(request, game_id)
    return manager.build_legal_actions(session)


@router.post("/{game_id}/action", response_model=GameStateResponse)
def submit_action(game_id: str, body: ActionRequest, request: Request):
    manager, _ = get_session(request, game_id)
    try:
        return manager.apply_human_action(game_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{game_id}/step", response_model=GameStateResponse)
def step_bots(game_id: str, request: Request):
    """Advance the game by running every pending bot turn (for all-bot games)."""
    manager, _ = get_session(request, game_id)
    return manager.step_bots(game_id)
