"""WebSocket endpoint for real-time game state push."""
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

log = logging.getLogger(__name__)

WS_GAME_NOT_FOUND = 4004


@router.websocket("/ws/{game_id}")
async def websocket_game(websocket: WebSocket, game_id: str):
    """
    Stream game state to a client.

    A WebSocket route cannot depend on `Request`, so the game manager is read
    off the application instance the socket is attached to.
    """
    manager = websocket.app.state.game_manager
    session = manager.get_session(game_id)
    if session is None:
        await websocket.close(code=WS_GAME_NOT_FOUND)
        return

    await websocket.accept()
    session.ws_connections.append(websocket)

    try:
        # Send current state immediately on connect
        await websocket.send_text(manager.build_state_response(session).model_dump_json())
        while True:
            # Any client message (ping/keepalive) triggers a fresh state push
            await websocket.receive_text()
            await websocket.send_text(manager.build_state_response(session).model_dump_json())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - never let a socket error kill the server
        log.exception("WebSocket error on game %s", game_id)
    finally:
        if websocket in session.ws_connections:
            session.ws_connections.remove(websocket)


async def broadcast_state(session, manager) -> None:
    """Push updated state to all connected WebSocket clients for a session."""
    if not session.ws_connections:
        return
    payload = manager.build_state_response(session).model_dump_json()
    dead = []
    for ws in session.ws_connections:
        try:
            await ws.send_text(payload)
        except Exception:  # noqa: BLE001 - drop clients that have gone away
            dead.append(ws)
    for ws in dead:
        if ws in session.ws_connections:
            session.ws_connections.remove(ws)
