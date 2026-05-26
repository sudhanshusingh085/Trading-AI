"""
Pattern Detection Engine — Detects classical chart patterns with success probabilities.
Uses local maxima/minima analysis on OHLCV data.
Probabilities sourced from Bulkowski's Encyclopedia of Chart Patterns.

NOTE: Patterns with <50% backtest accuracy have been removed to reduce false signals.
Removed: Double Bottom (41.7%), Ascending Triangle (37.5%), Rising Wedge (45.2%),
         Resistance Breakout (38.5%), Support Breakdown (38.1%)
"""

import numpy as np
from typing import List, Dict, Optional

def find_pivots(prices: List[float], window: int = 5) -> Dict[str, List[int]]:
    """Find local highs and lows (pivot points)."""
    highs = []
    lows = []
    
    for i in range(window, len(prices) - window):
        chunk = prices[i-window : i+window+1]
        if prices[i] == max(chunk):
            highs.append(i)
        if prices[i] == min(chunk):
            lows.append(i)
            
    return {"highs": highs, "lows": lows}

def detect_classical_patterns(candles: List[Dict]) -> List[Dict]:
    """
    Detects validated chart patterns that have >50% backtest accuracy.
    Returns list of signal dicts with historical probabilities.
    """
    # Filter out invalid candles
    valid_candles = [c for c in candles if c.get("close") is not None and c.get("high") is not None and c.get("low") is not None]
    if len(valid_candles) < 30:
        return []
    
    closes = [c["close"] for c in valid_candles]
    highs = [c["high"] for c in valid_candles]
    lows = [c["low"] for c in valid_candles]
    
    pivots = find_pivots(closes, window=7)
    signals = []
    
    latest_price = closes[-1]
    ts = valid_candles[-1]["timestamp"]
    
    # ─── DOUBLE BOTTOM — REMOVED (backtest accuracy: 41.7%) ───

    # ─── DOUBLE TOP (Prob: 73%, backtest accuracy: 52.9%) ───
    if len(pivots["highs"]) >= 2:
        h1_idx, h2_idx = pivots["highs"][-1], pivots["highs"][-2]
        h1_val, h2_val = closes[h1_idx], closes[h2_idx]
        diff = abs(h1_val - h2_val) / max(h1_val, h2_val)
        if diff < 0.02:
            hump_min = min(closes[h2_idx:h1_idx])
            if hump_min < h1_val * 0.97:
                signals.append(_psig("Double Top", "SELL", 4, 73, f"Resistance touch at {h1_val:.2f}", ts, latest_price))

    # ─── HEAD & SHOULDERS (Prob: 81%, backtest accuracy: 52.7%) ───
    if len(pivots["highs"]) >= 3:
        h1, h2, h3 = pivots["highs"][-3], pivots["highs"][-2], pivots["highs"][-1]
        v1, v2, v3 = closes[h1], closes[h2], closes[h3]
        if v2 > v1 and v2 > v3 and abs(v1-v3)/max(v1,v3) < 0.03:
            signals.append(_psig("Head and Shoulders", "SELL", 5, 81, "Classic bearish reversal pattern", ts, latest_price))

    # ─── INVERSE HEAD & SHOULDERS (Prob: 83%, backtest accuracy: 53.1%) ───
    if len(pivots["lows"]) >= 3:
        l1, l2, l3 = pivots["lows"][-3], pivots["lows"][-2], pivots["lows"][-1]
        v1, v2, v3 = closes[l1], closes[l2], closes[l3]
        if v2 < v1 and v2 < v3 and abs(v1-v3)/max(v1,v3) < 0.03:
            signals.append(_psig("Inverse Head and Shoulders", "BUY", 5, 83, "Strong bullish reversal pattern", ts, latest_price))

    # ─── ASCENDING TRIANGLE — REMOVED (backtest accuracy: 37.5%) ───
    # ─── RISING WEDGE — REMOVED (backtest accuracy: 45.2%) ───

    # ─── DESCENDING TRIANGLE (Prob: 72%, backtest accuracy: 60.7%) ───
    if len(pivots["highs"]) >= 2 and len(pivots["lows"]) >= 2:
        h_recent = [closes[i] for i in pivots["highs"][-3:]]
        l_recent = [closes[i] for i in pivots["lows"][-3:]]
        
        # Descending: Flat bottom, falling top
        is_flat_bottom = abs(l_recent[-1] - l_recent[-2])/l_recent[-1] < 0.01
        is_falling_top = h_recent[-1] < h_recent[-2] * 0.99
        if is_flat_bottom and is_falling_top:
            signals.append(_psig("Descending Triangle", "SELL", 4, 72, "Bearish consolidation / breakdown", ts, latest_price))

    # ─── RESISTANCE BREAKOUT — REMOVED (backtest accuracy: 38.5%) ───
    # ─── SUPPORT BREAKDOWN — REMOVED (backtest accuracy: 38.1%) ───

    return signals

def _psig(name, direction, strength, probability, reason, ts, price):
    return {
        "name": name,
        "type": "chart",
        "direction": direction,
        "strength": strength,
        "probability": probability,
        "reason": reason,
        "timestamp": ts,
        "price": price
    }
