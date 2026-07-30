"""WebSocket endpoint for real-time game state push."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
import json

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str, request: Request):
    manager = request.app.state.game_manager
    session = manager.get_session(game_id)
    if session is None:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    session.ws_connections.append(websocket)

    # Send current state immediately on connect
    state_response = manager._build_response(session)
    await websocket.send_text(state_response.model_dump_json())

    try:
        while True:
            # Wait for any message from client (ping/keepalive)
            data = await websocket.receive_text()
            # Echo back current state on any client message
            state_response = manager._build_response(session)
            await websocket.send_text(state_response.model_dump_json())
    except WebSocketDisconnect:
        session.ws_connections.remove(websocket)


async def broadcast_state(session, manager) -> None:
    """Push updated state to all connected WebSocket clients for a session."""
    if not session.ws_connections:
        return
    state_response = manager._build_response(session)
    payload = state_response.model_dump_json()
    dead = []
    for ws in session.ws_connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        session.ws_connections.remove(ws)
