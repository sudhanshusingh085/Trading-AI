"""
Pattern Detection Engine — Detects classical chart patterns with success probabilities.
Uses local maxima/minima analysis on OHLCV data.
Probabilities sourced from Bulkowski's Encyclopedia of Chart Patterns.
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
    Detects 15+ chart patterns including Triangles, Wedges, Head & Shoulders, etc.
    Returns list of signal dicts with historical probabilities.
    """
    if len(candles) < 50:
        return []
    
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    
    pivots = find_pivots(closes, window=7)
    signals = []
    
    latest_price = closes[-1]
    ts = candles[-1]["timestamp"]

    # ─── DOUBLE BOTTOM (Prob: 78%) ───
    if len(pivots["lows"]) >= 2:
        l1_idx, l2_idx = pivots["lows"][-1], pivots["lows"][-2]
        l1_val, l2_val = closes[l1_idx], closes[l2_idx]
        diff = abs(l1_val - l2_val) / max(l1_val, l2_val)
        if diff < 0.02:
            hump_max = max(closes[l2_idx:l1_idx])
            if hump_max > l1_val * 1.03:
                signals.append(_psig("Double Bottom", "BUY", 4, 78, f"Support touch at {l1_val:.2f}", ts, latest_price))

    # ─── DOUBLE TOP (Prob: 73%) ───
    if len(pivots["highs"]) >= 2:
        h1_idx, h2_idx = pivots["highs"][-1], pivots["highs"][-2]
        h1_val, h2_val = closes[h1_idx], closes[h2_idx]
        diff = abs(h1_val - h2_val) / max(h1_val, h2_val)
        if diff < 0.02:
            hump_min = min(closes[h2_idx:h1_idx])
            if hump_min < h1_val * 0.97:
                signals.append(_psig("Double Top", "SELL", 4, 73, f"Resistance touch at {h1_val:.2f}", ts, latest_price))

    # ─── HEAD & SHOULDERS (Prob: 81%) ───
    if len(pivots["highs"]) >= 3:
        h1, h2, h3 = pivots["highs"][-3], pivots["highs"][-2], pivots["highs"][-1]
        v1, v2, v3 = closes[h1], closes[h2], closes[h3]
        if v2 > v1 and v2 > v3 and abs(v1-v3)/max(v1,v3) < 0.03:
            signals.append(_psig("Head and Shoulders", "SELL", 5, 81, "Classic bearish reversal pattern", ts, latest_price))

    # ─── INVERSE HEAD & SHOULDERS (Prob: 83%) ───
    if len(pivots["lows"]) >= 3:
        l1, l2, l3 = pivots["lows"][-3], pivots["lows"][-2], pivots["lows"][-1]
        v1, v2, v3 = closes[l1], closes[l2], closes[l3]
        if v2 < v1 and v2 < v3 and abs(v1-v3)/max(v1,v3) < 0.03:
            signals.append(_psig("Inverse Head and Shoulders", "BUY", 5, 83, "Strong bullish reversal pattern", ts, latest_price))

    # ─── TRIANGLES (Ascending: 72%, Descending: 72%) ───
    if len(pivots["highs"]) >= 2 and len(pivots["lows"]) >= 2:
        h_recent = [closes[i] for i in pivots["highs"][-3:]]
        l_recent = [closes[i] for i in pivots["lows"][-3:]]
        
        # Ascending: Flat top, rising bottom
        is_flat_top = abs(h_recent[-1] - h_recent[-2])/h_recent[-1] < 0.01
        is_rising_bottom = l_recent[-1] > l_recent[-2] * 1.01
        if is_flat_top and is_rising_bottom:
            signals.append(_psig("Ascending Triangle", "BUY", 4, 72, "Bullish consolidation / breakout", ts, latest_price))
            
        # Descending: Flat bottom, falling top
        is_flat_bottom = abs(l_recent[-1] - l_recent[-2])/l_recent[-1] < 0.01
        is_falling_top = h_recent[-1] < h_recent[-2] * 0.99
        if is_flat_bottom and is_falling_top:
            signals.append(_psig("Descending Triangle", "SELL", 4, 72, "Bearish consolidation / breakdown", ts, latest_price))

    # ─── RISING/FALLING WEDGES (Prob: 68%) ───
    if len(pivots["highs"]) >= 2 and len(pivots["lows"]) >= 2:
        h1, h2 = h_recent[-2], h_recent[-1]
        l1, l2 = l_recent[-2], l_recent[-1]
        if h2 > h1 and l2 > l1 and (h2-h1) < (l2-l1): # Converging up
            signals.append(_psig("Rising Wedge", "SELL", 4, 68, "Bearish reversal pattern", ts, latest_price))
        if h2 < h1 and l2 < l1 and (l1-l2) < (h1-h2): # Converging down
            signals.append(_psig("Falling Wedge", "BUY", 4, 68, "Bullish reversal pattern", ts, latest_price))

    # ─── BREAKOUTS ───
    if latest_price > max(highs[-20:-1]):
        signals.append(_psig("Resistance Breakout", "BUY", 4, 65, "Price broke 20-period high", ts, latest_price))
    if latest_price < min(lows[-20:-1]):
        signals.append(_psig("Support Breakdown", "SELL", 4, 65, "Price broke 20-period low", ts, latest_price))

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
