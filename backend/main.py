"""
Trading AI — FastAPI Backend Entry Point
Serves REST API + WebSocket for live data streaming.
Starts background scanner on launch.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import os

from backend.api.routes import router
from backend.signals.scanner import scanner
from backend.data.crypto_fetcher import stream_live_price


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background scanner on app startup."""
    print("[*] Trading AI Backend starting...")
    print("[*] Starting continuous market scanner...")
    scan_task = asyncio.create_task(scanner.start_continuous_scan())
    yield
    print("[*] Shutting down scanner...")
    scanner.stop()
    scan_task.cancel()


app = FastAPI(
    title="Trading AI",
    description="Data-driven trading analysis platform for Indian Stocks & Crypto",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)

# Serve frontend static files if built
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok", "scanner_running": scanner.is_running}


@app.websocket("/ws/crypto/{symbol}")
async def crypto_live_ws(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for live crypto price streaming."""
    await websocket.accept()
    try:
        async for price_data in stream_live_price(symbol):
            await websocket.send_json(price_data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/scanner")
async def scanner_ws(websocket: WebSocket):
    """WebSocket for real-time scanner updates."""
    await websocket.accept()
    scanner.subscribe(websocket)
    try:
        while True:
            # Keep connection alive, listen for config updates
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "set_interval":
                scanner.scan_interval = msg.get("seconds", 120)
    except WebSocketDisconnect:
        scanner.unsubscribe(websocket)
    except Exception:
        scanner.unsubscribe(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
