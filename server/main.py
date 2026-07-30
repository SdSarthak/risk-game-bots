"""
FastAPI application entry point.

Run with:
  uvicorn server.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.game_manager import GameManager
from server.routers import games, ws

app = FastAPI(
    title="Risk Game Bots API",
    description="Play Risk against AI bots or watch them compete.",
    version="1.0.0",
)

# Allow the React frontend (default Vite port 5173) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach game manager as app state (shared across requests)
app.state.game_manager = GameManager()

# Routers
app.include_router(games.router)
app.include_router(ws.router)


@app.get("/")
def root():
    return {"message": "Risk Game Bots API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
