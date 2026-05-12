"""
Real-Time Market Scanner — Continuously scans all tracked assets for signals.
Runs as a background async task, pushing alerts to connected clients.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from backend.data.stock_fetcher import get_historical_data as get_stock_data, NIFTY_50
from backend.data.crypto_fetcher import get_historical_data_sync as get_crypto_data, TOP_CRYPTO_PAIRS
from backend.indicators.calculator import compute_all_indicators
from backend.signals.detector import detect_signals


class MarketScanner:
    """Continuously scans markets and maintains a live signal board."""

    def __init__(self):
        self.scan_results = {}  # symbol -> latest scan result
        self.active_signals = []  # all active signals across markets
        self.last_scan_time = None
        self.is_running = False
        self.scan_interval = 120  # seconds between full scans
        self.subscribers = set()  # WebSocket connections to notify

    def subscribe(self, ws):
        self.subscribers.add(ws)

    def unsubscribe(self, ws):
        self.subscribers.discard(ws)

    async def notify_subscribers(self, data: dict):
        dead = set()
        for ws in self.subscribers:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.subscribers -= dead

    def scan_symbol_stock(self, symbol: str, interval: str = "1d", period: str = "6mo") -> dict:
        """Scan a single Indian stock for signals."""
        data = get_stock_data(symbol, interval=interval, period=period)
        if not data or "error" in data:
            return {"symbol": symbol, "market": "stock", "error": data.get("error", "No data")}

        candles = data["candles"]
        if len(candles) < 30:
            return {"symbol": symbol, "market": "stock", "error": "Insufficient data"}

        indicators = compute_all_indicators(candles)
        signals = detect_signals(candles, indicators)

        result = {
            "symbol": symbol,
            "market": "stock",
            "price": data["latest_price"],
            "interval": interval,
            "signals": signals,
            "signal_count": len(signals),
            "buy_signals": len([s for s in signals if s["direction"] == "BUY"]),
            "sell_signals": len([s for s in signals if s["direction"] == "SELL"]),
            "buy_confluence": signals[0]["buy_confluence"] if signals else 0,
            "sell_confluence": signals[0]["sell_confluence"] if signals else 0,
            "verdict": _get_verdict(signals),
            "scanned_at": datetime.now().isoformat()
        }
        return result

    def scan_symbol_crypto(self, symbol: str, interval: str = "1d", limit: int = 200) -> dict:
        """Scan a single crypto pair for signals."""
        data = get_crypto_data(symbol, interval=interval, limit=limit)
        if not data or "error" in data:
            return {"symbol": symbol, "market": "crypto", "error": data.get("error", "No data")}

        candles = data["candles"]
        if len(candles) < 30:
            return {"symbol": symbol, "market": "crypto", "error": "Insufficient data"}

        indicators = compute_all_indicators(candles)
        signals = detect_signals(candles, indicators)

        result = {
            "symbol": symbol,
            "market": "crypto",
            "price": data["latest_price"],
            "interval": interval,
            "signals": signals,
            "signal_count": len(signals),
            "buy_signals": len([s for s in signals if s["direction"] == "BUY"]),
            "sell_signals": len([s for s in signals if s["direction"] == "SELL"]),
            "buy_confluence": signals[0]["buy_confluence"] if signals else 0,
            "sell_confluence": signals[0]["sell_confluence"] if signals else 0,
            "verdict": _get_verdict(signals),
            "scanned_at": datetime.now().isoformat()
        }
        return result

    async def run_full_scan(self, stock_interval="1d", crypto_interval="1d"):
        """Run a full scan across all tracked assets."""
        self.active_signals = []
        all_results = []

        # Scan NIFTY 50 stocks
        for symbol in NIFTY_50[:20]:  # Start with top 20 to be fast
            try:
                result = self.scan_symbol_stock(symbol, interval=stock_interval)
                self.scan_results[symbol] = result
                all_results.append(result)
                if result.get("signals"):
                    for sig in result["signals"]:
                        sig["symbol"] = symbol
                        sig["market"] = "stock"
                        self.active_signals.append(sig)
            except Exception as e:
                pass
            await asyncio.sleep(0.5)  # Rate limiting

        # Scan top crypto pairs
        for symbol in TOP_CRYPTO_PAIRS[:20]:  # Top 20
            try:
                result = self.scan_symbol_crypto(symbol, interval=crypto_interval)
                self.scan_results[symbol] = result
                all_results.append(result)
                if result.get("signals"):
                    for sig in result["signals"]:
                        sig["symbol"] = symbol
                        sig["market"] = "crypto"
                        self.active_signals.append(sig)
            except Exception as e:
                pass
            await asyncio.sleep(0.3)

        # Sort active signals by strength
        self.active_signals.sort(key=lambda x: x.get("strength", 0), reverse=True)
        self.last_scan_time = datetime.now().isoformat()

        # Notify subscribers
        await self.notify_subscribers({
            "type": "scan_complete",
            "total_scanned": len(all_results),
            "total_signals": len(self.active_signals),
            "top_signals": self.active_signals[:20],
            "results": [r for r in all_results if r.get("signal_count", 0) > 0],
            "scanned_at": self.last_scan_time
        })

        return all_results

    async def start_continuous_scan(self):
        """Run scanner continuously in the background."""
        self.is_running = True
        while self.is_running:
            try:
                await self.run_full_scan()
            except Exception as e:
                print(f"[Scanner] Error during scan: {e}")
            await asyncio.sleep(self.scan_interval)

    def stop(self):
        self.is_running = False

    def get_top_opportunities(self, limit: int = 10, direction: str = None) -> list:
        """Get top trading opportunities from the latest scan."""
        signals = self.active_signals
        if direction:
            signals = [s for s in signals if s["direction"] == direction.upper()]
        return signals[:limit]

    def get_scan_summary(self) -> dict:
        """Get a summary of the latest scan."""
        return {
            "last_scan": self.last_scan_time,
            "total_assets_scanned": len(self.scan_results),
            "total_active_signals": len(self.active_signals),
            "buy_signals": len([s for s in self.active_signals if s["direction"] == "BUY"]),
            "sell_signals": len([s for s in self.active_signals if s["direction"] == "SELL"]),
            "top_buys": self.get_top_opportunities(5, "BUY"),
            "top_sells": self.get_top_opportunities(5, "SELL"),
            "is_running": self.is_running,
            "scan_interval_seconds": self.scan_interval,
        }


def _get_verdict(signals: list) -> str:
    """Determine overall verdict from signals."""
    if not signals:
        return "NEUTRAL"
    buy_str = sum(s["strength"] for s in signals if s["direction"] == "BUY")
    sell_str = sum(s["strength"] for s in signals if s["direction"] == "SELL")
    if buy_str > sell_str + 3:
        return "STRONG BUY"
    elif buy_str > sell_str:
        return "BUY"
    elif sell_str > buy_str + 3:
        return "STRONG SELL"
    elif sell_str > buy_str:
        return "SELL"
    return "NEUTRAL"


# Global scanner instance
scanner = MarketScanner()
