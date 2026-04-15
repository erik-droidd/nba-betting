"""FastAPI application for the NBA betting system."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from nba_betting.api.routes import router

app = FastAPI(
    title="NBA Betting System",
    description="NBA game predictions with Polymarket odds comparison",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve the dashboard
STATIC_DIR = Path(__file__).parent.parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return HTMLResponse(
        "<h1>NBA Betting System</h1><p>Dashboard not found. API available at /api/</p>"
    )
