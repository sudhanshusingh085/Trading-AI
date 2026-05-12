"""
Stock Data Fetcher — Indian Stocks (NSE/BSE) via yfinance
Handles historical OHLCV data with caching to minimize API calls.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import json
import os

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "stocks")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── NIFTY 50 Constituents ───
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "NTPC", "TITAN", "BAJFINANCE",
    "BAJAJFINSV", "WIPRO", "ONGC", "JSWSTEEL", "POWERGRID",
    "M&M", "TATASTEEL", "ADANIENT", "ADANIPORTS", "ULTRACEMCO",
    "TECHM", "NESTLEIND", "INDUSINDBK", "DIVISLAB", "GRASIM",
    "DRREDDY", "CIPLA", "COALINDIA", "BPCL", "APOLLOHOSP",
    "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "SBILIFE", "TATACONSUM",
    "BAJAJ-AUTO", "HINDALCO", "HDFCLIFE", "UPL", "LTIM"
]

# ─── Popular Indian Stocks (beyond NIFTY 50) ───
POPULAR_STOCKS = [
    "ZOMATO", "PAYTM", "NYKAA", "DELHIVERY", "IRCTC",
    "TATAPOWER", "ADANIGREEN", "VEDL", "BANKBARODA", "PNB",
    "IDEA", "SAIL", "NHPC", "IOC", "GAIL",
    "HAL", "BEL", "BHEL", "IRFC", "RECLTD"
]

ALL_STOCKS = NIFTY_50 + POPULAR_STOCKS

# Interval mapping for yfinance
VALID_INTERVALS = {
    "1m": {"max_period": "7d", "description": "1 minute"},
    "2m": {"max_period": "60d", "description": "2 minutes"},
    "5m": {"max_period": "60d", "description": "5 minutes"},
    "15m": {"max_period": "60d", "description": "15 minutes"},
    "30m": {"max_period": "60d", "description": "30 minutes"},
    "1h": {"max_period": "730d", "description": "1 hour"},
    "1d": {"max_period": "max", "description": "1 day"},
    "1wk": {"max_period": "max", "description": "1 week"},
}


def _get_cache_key(symbol: str, interval: str, period: str) -> str:
    """Generate a cache filename."""
    return f"{symbol}_{interval}_{period}_{datetime.now().strftime('%Y%m%d')}.parquet"


def _format_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Add exchange suffix to symbol."""
    symbol = symbol.upper().strip()
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        return symbol + suffix
    return symbol


def get_historical_data(
    symbol: str,
    interval: str = "1d",
    period: str = "1y",
    exchange: str = "NSE",
    use_cache: bool = True
) -> Optional[dict]:
    """
    Fetch historical OHLCV data for an Indian stock.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS")
        interval: Candle interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        exchange: NSE or BSE
        use_cache: Whether to use cached data
    
    Returns:
        Dict with OHLCV data and metadata, or None on error
    """
    if interval not in VALID_INTERVALS:
        return {"error": f"Invalid interval. Valid: {list(VALID_INTERVALS.keys())}"}
    
    formatted_symbol = _format_symbol(symbol, exchange)
    
    # Check cache
    cache_key = _get_cache_key(formatted_symbol, interval, period)
    cache_path = os.path.join(CACHE_DIR, cache_key)
    
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return _format_response(df, symbol, formatted_symbol, interval)
        except Exception:
            pass
    
    try:
        ticker = yf.Ticker(formatted_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Clean up columns
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        # Rename date/datetime column
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        
        # Ensure timestamp is string for JSON serialization
        df["timestamp"] = df["timestamp"].astype(str)
        
        # Cache the data
        if use_cache:
            try:
                df.to_parquet(cache_path, index=False)
            except Exception:
                pass
        
        return _format_response(df, symbol, formatted_symbol, interval)
        
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}


def _format_response(df: pd.DataFrame, symbol: str, formatted_symbol: str, interval: str) -> dict:
    """Format DataFrame into API response."""
    # Select only the columns we need
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    available_cols = [c for c in required_cols if c in df.columns]
    df_clean = df[available_cols].copy()
    
    # Convert to list of dicts for JSON
    candles = df_clean.to_dict(orient="records")
    
    # Replace NaN with None
    for candle in candles:
        for key in candle:
            if pd.isna(candle[key]):
                candle[key] = None
    
    return {
        "symbol": symbol,
        "formatted_symbol": formatted_symbol,
        "interval": interval,
        "candle_count": len(candles),
        "candles": candles,
        "latest_price": candles[-1]["close"] if candles else None,
        "latest_volume": candles[-1]["volume"] if candles else None,
    }


def search_stocks(query: str) -> list:
    """
    Search for Indian stock symbols matching a query.
    
    Args:
        query: Search string (partial symbol or name)
    
    Returns:
        List of matching stock symbols
    """
    query = query.upper().strip()
    matches = []
    
    for stock in ALL_STOCKS:
        if query in stock:
            matches.append({
                "symbol": stock,
                "exchange": "NSE",
                "formatted": f"{stock}.NS"
            })
    
    return matches[:20]  # Limit results


def get_stock_info(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """Get basic info about a stock."""
    formatted_symbol = _format_symbol(symbol, exchange)
    
    try:
        ticker = yf.Ticker(formatted_symbol)
        info = ticker.info
        
        return {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
        }
    except Exception as e:
        return {"error": str(e)}
