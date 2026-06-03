"""
Crypto Data Fetcher — via Binance REST API and WebSocket
Handles historical OHLCV data and real-time price streaming.
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import Optional, AsyncGenerator
import json
import os
import websockets

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "crypto")
os.makedirs(CACHE_DIR, exist_ok=True)

# Binance API base
BINANCE_BASE = "https://api.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws"

# ─── Top Crypto Pairs ───
TOP_CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
    "SHIBUSDT", "LTCUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT",
    "XLMUSDT", "ETCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "FILUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
    "PEPEUSDT", "WIFUSDT", "FETUSDT", "RENDERUSDT", "TIAUSDT",
    "JUPUSDT", "STXUSDT", "IMXUSDT", "GRTUSDT", "AAVEUSDT",
    "MKRUSDT", "SNXUSDT", "RUNEUSDT", "PENDLEUSDT", "ENAUSDT",
    "ONDOUSDT", "WLDUSDT", "ORDIUSDT", "KASUSDT", "TONUSDT",
    "TRXUSDT", "BCHUSDT", "ICPUSDT", "HBARUSDT", "VETUSDT"
]

# ─── Indian-focused pairs (INR pairs on WazirX/Binance) ───
INR_PAIRS = [
    "BTCINR", "ETHINR", "BNBINR", "XRPINR", "DOGEINR",
    "SOLINR", "MATICINR", "ADAINR"
]

# Binance interval mapping
VALID_INTERVALS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
    "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
}


async def get_historical_data(
    symbol: str,
    interval: str = "1d",
    limit: int = 500
) -> Optional[dict]:
    """
    Fetch historical OHLCV data for a crypto pair from Binance.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
        limit: Number of candles to fetch (max 1000)
    
    Returns:
        Dict with OHLCV data and metadata
    """
    symbol = symbol.upper().strip()
    bi_interval = VALID_INTERVALS.get(interval, interval)
    limit = min(limit, 1000)
    
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": bi_interval,
        "limit": limit
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {"error": f"Binance API error: {error_text}"}
                
                data = await resp.json()
        
        candles = []
        for kline in data:
            candles.append({
                "timestamp": datetime.fromtimestamp(kline[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_volume": float(kline[7]),
                "trades": int(kline[8])
            })
        
        return {
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(candles),
            "candles": candles,
            "latest_price": candles[-1]["close"] if candles else None,
            "latest_volume": candles[-1]["volume"] if candles else None,
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch crypto data: {str(e)}"}


def get_historical_data_sync(
    symbol: str,
    interval: str = "1d",
    limit: int = 500
) -> Optional[dict]:
    """Synchronous wrapper for get_historical_data."""
    import requests
    
    symbol = symbol.upper().strip()
    bi_interval = VALID_INTERVALS.get(interval, interval)
    limit = min(limit, 1000)
    
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": bi_interval,
        "limit": limit
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            return {"error": f"Binance API error: {resp.text}"}
        
        data = resp.json()
        
        candles = []
        for kline in data:
            candles.append({
                "timestamp": datetime.fromtimestamp(kline[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_volume": float(kline[7]),
                "trades": int(kline[8])
            })
        
        return {
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(candles),
            "candles": candles,
            "latest_price": candles[-1]["close"] if candles else None,
            "latest_volume": candles[-1]["volume"] if candles else None,
        }
    except Exception as e:
        return {"error": f"Failed to fetch crypto data: {str(e)}"}


def get_historical_data_sync_large(
    symbol: str,
    interval: str = "1h",
    total_candles: int = 5000
) -> Optional[dict]:
    """
    Paginated sync fetcher — overcomes Binance's 1000-candle-per-call limit.
    Chains multiple API calls walking backwards in time, then merges chronologically.

    Args:
        symbol:        Trading pair (e.g., "BTCUSDT")
        interval:      Candle interval (1m, 5m, 15m, 1h, 4h, 1d …)
        total_candles: How many candles to fetch in total (max ~5000 recommended)

    Returns:
        Same dict shape as get_historical_data_sync.
    """
    import requests
    import time as _time

    symbol      = symbol.upper().strip()
    bi_interval = VALID_INTERVALS.get(interval, interval)
    chunk_size  = 1000          # Binance hard limit per call
    url         = f"{BINANCE_BASE}/api/v3/klines"

    all_klines = []
    end_time   = None           # Walk backwards using endTime

    while len(all_klines) < total_candles:
        remaining = total_candles - len(all_klines)
        fetch_n   = min(chunk_size, remaining)

        params = {
            "symbol":   symbol,
            "interval": bi_interval,
            "limit":    fetch_n,
        }
        if end_time is not None:
            params["endTime"] = end_time  # fetch candles ending before this timestamp (ms)

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"[-] Binance error ({symbol}): {resp.text}")
                break
            chunk = resp.json()
        except Exception as e:
            print(f"[-] Request failed ({symbol}): {e}")
            break

        if not chunk:
            break

        # Prepend (we're walking backwards)
        all_klines = chunk + all_klines

        # Move end_time to just before the earliest candle in this chunk
        end_time = chunk[0][0] - 1  # open_time of oldest candle minus 1 ms

        if len(chunk) < fetch_n:
            break   # No more history available

        _time.sleep(0.15)  # Respect Binance rate limit (~6 req/s)

    if not all_klines:
        return {"error": f"No data returned for {symbol}"}

    candles = []
    for kline in all_klines:
        candles.append({
            "timestamp":    datetime.fromtimestamp(kline[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            "open":         float(kline[1]),
            "high":         float(kline[2]),
            "low":          float(kline[3]),
            "close":        float(kline[4]),
            "volume":       float(kline[5]),
            "quote_volume": float(kline[7]),
            "trades":       int(kline[8]),
        })

    print(f"[+] Fetched {len(candles)} candles for {symbol} @ {interval}")
    return {
        "symbol":        symbol,
        "interval":      interval,
        "candle_count":  len(candles),
        "candles":       candles,
        "latest_price":  candles[-1]["close"] if candles else None,
        "latest_volume": candles[-1]["volume"] if candles else None,
    }


async def stream_live_price(symbol: str) -> AsyncGenerator[dict, None]:
    """
    Stream real-time price updates via Binance WebSocket.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
    
    Yields:
        Dict with current price data
    """
    symbol_lower = symbol.lower().strip()
    ws_url = f"{BINANCE_WS}/{symbol_lower}@trade"
    
    try:
        async with websockets.connect(ws_url) as ws:
            async for message in ws:
                data = json.loads(message)
                yield {
                    "symbol": data["s"],
                    "price": float(data["p"]),
                    "quantity": float(data["q"]),
                    "timestamp": datetime.fromtimestamp(data["T"] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_buyer_maker": data["m"]
                }
    except Exception as e:
        yield {"error": f"WebSocket error: {str(e)}"}


async def get_ticker_24h(symbol: str = None) -> Optional[dict]:
    """
    Get 24h ticker data for a symbol or all symbols.
    """
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
        
        if isinstance(data, list):
            # Filter to only our tracked pairs
            filtered = []
            for t in data:
                if t["symbol"] in TOP_CRYPTO_PAIRS:
                    filtered.append({
                        "symbol": t["symbol"],
                        "price": float(t["lastPrice"]),
                        "change_percent": float(t["priceChangePercent"]),
                        "high": float(t["highPrice"]),
                        "low": float(t["lowPrice"]),
                        "volume": float(t["volume"]),
                        "quote_volume": float(t["quoteVolume"]),
                    })
            return {"tickers": filtered}
        else:
            return {
                "symbol": data["symbol"],
                "price": float(data["lastPrice"]),
                "change_percent": float(data["priceChangePercent"]),
                "high": float(data["highPrice"]),
                "low": float(data["lowPrice"]),
                "volume": float(data["volume"]),
                "quote_volume": float(data["quoteVolume"]),
            }
    except Exception as e:
        return {"error": str(e)}


def search_crypto(query: str) -> list:
    """Search for crypto pairs matching a query."""
    query = query.upper().strip()
    matches = []
    
    for pair in TOP_CRYPTO_PAIRS:
        if query in pair:
            base = pair.replace("USDT", "")
            matches.append({
                "symbol": pair,
                "base": base,
                "quote": "USDT",
                "display": f"{base}/USDT"
            })
    
    return matches[:20]
