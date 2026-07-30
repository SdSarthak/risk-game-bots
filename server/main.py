"""
FastAPI application entry point.

Run with:
  uvicorn server.main:app --reload --port 8000

Configuration is read from the environment (see .env.example):
  RISK_CORS_ORIGINS  comma-separated browser origins allowed to call the API
  RISK_MAX_SESSIONS  how many concurrent games to keep in memory
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.game_manager import GameManager
from server.routers import games, ws

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"


def cors_origins() -> list[str]:
    """Allowed browser origins, from RISK_CORS_ORIGINS or the Vite/CRA defaults."""
    raw = os.getenv("RISK_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def max_sessions() -> int:
    raw = os.getenv("RISK_MAX_SESSIONS", "200")
    try:
        return max(1, int(raw))
    except ValueError:
        return 200


def create_app() -> FastAPI:
    app = FastAPI(
        title="Risk Game Bots API",
        description="Play Risk against AI bots or watch them compete.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared across requests; games live only for the lifetime of the process
    app.state.game_manager = GameManager(max_sessions=max_sessions())

    app.include_router(games.router)
    app.include_router(ws.router)

    @app.get("/")
    def root():
        return {"message": "Risk Game Bots API", "docs": "/docs"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
