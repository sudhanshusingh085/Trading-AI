"""
API Routes — REST endpoints for the Trading AI platform.
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from typing import Optional
import asyncio

from backend.data.stock_fetcher import (
    get_historical_data as get_stock_history,
    search_stocks, get_stock_info, NIFTY_50, ALL_STOCKS
)
from backend.data.crypto_fetcher import (
    get_historical_data_sync as get_crypto_history,
    search_crypto, TOP_CRYPTO_PAIRS
)
from backend.indicators.calculator import compute_all_indicators, get_available_indicators
from backend.signals.detector import detect_signals
from backend.signals.scanner import scanner
from backend.backtest.engine import run_backtest
from backend.backtest.strategies import STRATEGIES

router = APIRouter()


# ═══════════════════════════════════════════
#  STOCK ENDPOINTS
# ═══════════════════════════════════════════

@router.get("/api/stocks/list")
async def list_stocks():
    return {"stocks": ALL_STOCKS, "nifty_50": NIFTY_50}


@router.get("/api/stocks/search")
async def stock_search(q: str = Query(...)):
    return {"results": search_stocks(q)}


@router.get("/api/stocks/{symbol}/info")
async def stock_info(symbol: str):
    return get_stock_info(symbol)


@router.get("/api/stocks/{symbol}/history")
async def stock_history(
    symbol: str,
    interval: str = Query("1d"),
    period: str = Query("6mo"),
):
    data = get_stock_history(symbol, interval=interval, period=period)
    if not data or "error" in data:
        return data or {"error": "No data"}
    return data


@router.get("/api/stocks/{symbol}/analysis")
async def stock_analysis(
    symbol: str,
    interval: str = Query("1d"),
    period: str = Query("6mo"),
):
    """Full analysis: history + indicators + signals."""
    data = get_stock_history(symbol, interval=interval, period=period)
    if not data or "error" in data:
        return data or {"error": "No data"}

    indicators = compute_all_indicators(data["candles"])
    signals = detect_signals(data["candles"], indicators)

    return {
        **data,
        "indicators": indicators,
        "signals": signals,
        "verdict": _verdict(signals),
    }


# ═══════════════════════════════════════════
#  CRYPTO ENDPOINTS
# ═══════════════════════════════════════════

@router.get("/api/crypto/pairs")
async def list_crypto():
    return {"pairs": TOP_CRYPTO_PAIRS}


@router.get("/api/crypto/search")
async def crypto_search(q: str = Query(...)):
    return {"results": search_crypto(q)}


@router.get("/api/crypto/{symbol}/history")
async def crypto_history(
    symbol: str,
    interval: str = Query("1d"),
    limit: int = Query(500),
):
    data = get_crypto_history(symbol, interval=interval, limit=limit)
    if not data or "error" in data:
        return data or {"error": "No data"}
    return data


@router.get("/api/crypto/{symbol}/analysis")
async def crypto_analysis(
    symbol: str,
    interval: str = Query("1d"),
    limit: int = Query(500),
):
    data = get_crypto_history(symbol, interval=interval, limit=limit)
    if not data or "error" in data:
        return data or {"error": "No data"}

    indicators = compute_all_indicators(data["candles"])
    signals = detect_signals(data["candles"], indicators)

    return {
        **data,
        "indicators": indicators,
        "signals": signals,
        "verdict": _verdict(signals),
    }


@router.get("/api/crypto/{symbol}/multi-tf")
async def crypto_multi_timeframe(
    symbol: str,
    timeframes: str = Query("1m,5m,15m,1h,4h,1d"),
):
    """Fetch verdicts for multiple timeframes in one call."""
    tf_list = [tf.strip() for tf in timeframes.split(",") if tf.strip()]
    results = {}
    for tf in tf_list:
        try:
            limit = 200 if tf in ("1m", "3m", "5m") else 300
            data = get_crypto_history(symbol, interval=tf, limit=limit)
            if data and "error" not in data and len(data.get("candles", [])) >= 30:
                indicators = compute_all_indicators(data["candles"])
                signals = detect_signals(data["candles"], indicators)
                buy_str = sum(s["strength"] for s in signals if s["direction"] == "BUY")
                sell_str = sum(s["strength"] for s in signals if s["direction"] == "SELL")
                results[tf] = {
                    "verdict": _verdict(signals),
                    "signal_count": len(signals),
                    "buy_strength": buy_str,
                    "sell_strength": sell_str,
                    "top_signal": signals[0]["name"] if signals else None,
                }
            else:
                results[tf] = {"verdict": "NO DATA", "signal_count": 0, "buy_strength": 0, "sell_strength": 0, "top_signal": None}
        except Exception:
            results[tf] = {"verdict": "ERROR", "signal_count": 0, "buy_strength": 0, "sell_strength": 0, "top_signal": None}
    return {"symbol": symbol, "timeframes": results}


# ═══════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════

@router.get("/api/indicators/available")
async def available_indicators():
    return {"indicators": get_available_indicators()}


# ═══════════════════════════════════════════
#  SCANNER
# ═══════════════════════════════════════════

@router.get("/api/scanner/summary")
async def scanner_summary():
    return scanner.get_scan_summary()


@router.get("/api/scanner/opportunities")
async def scanner_opportunities(
    direction: Optional[str] = None,
    limit: int = Query(20),
):
    return {"opportunities": scanner.get_top_opportunities(limit, direction)}


@router.post("/api/scanner/run")
async def run_scan():
    """Trigger a manual scan."""
    results = await scanner.run_full_scan()
    return {"scanned": len(results), "summary": scanner.get_scan_summary()}


@router.get("/api/scanner/results")
async def scanner_results():
    return {"results": scanner.scan_results, "signals": scanner.active_signals[:50]}


# ═══════════════════════════════════════════
#  BACKTESTING
# ═══════════════════════════════════════════

@router.get("/api/backtest/strategies")
async def list_strategies():
    return {"strategies": {k: {kk: vv for kk, vv in v.items() if kk != "fn"} for k, v in STRATEGIES.items()}}


@router.post("/api/backtest/run")
async def backtest_run(
    symbol: str = Query(...),
    market: str = Query("stock"),
    strategy: str = Query("ema_9_21"),
    interval: str = Query("1d"),
    period: str = Query("1y"),
    initial_capital: float = Query(100000),
    stop_loss: float = Query(2.0),
    take_profit: float = Query(5.0),
):
    if strategy not in STRATEGIES:
        return {"error": f"Unknown strategy. Available: {list(STRATEGIES.keys())}"}

    if market == "stock":
        data = get_stock_history(symbol, interval=interval, period=period)
    else:
        data = get_crypto_history(symbol, interval=interval, limit=500)

    if not data or "error" in data:
        return data or {"error": "No data"}

    result = run_backtest(
        candles=data["candles"],
        strategy_fn=STRATEGIES[strategy]["fn"],
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    return {
        "symbol": symbol,
        "strategy": STRATEGIES[strategy]["name"],
        **result,
    }


def _verdict(signals):
    if not signals:
        return "NEUTRAL"
    buy = sum(s["strength"] for s in signals if s["direction"] == "BUY")
    sell = sum(s["strength"] for s in signals if s["direction"] == "SELL")
    if buy > sell + 3: return "STRONG BUY"
    if buy > sell: return "BUY"
    if sell > buy + 3: return "STRONG SELL"
    if sell > buy: return "SELL"
    return "NEUTRAL"
