"""
Pattern Detection Engine — Detects classical chart patterns (Double Top/Bottom, Breakouts).
Uses local maxima/minima analysis on OHLCV data.
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
    Detect Double Top, Double Bottom, and Breakouts.
    Returns list of signal dicts.
    """
    # Filter out invalid candles
    valid_candles = [c for c in candles if c.get("close") is not None and c.get("high") is not None and c.get("low") is not None]
    if len(valid_candles) < 30:
        return []
    
    closes = [c["close"] for c in valid_candles]
    highs = [c["high"] for c in valid_candles]
    lows = [c["low"] for c in valid_candles]
    
    pivots = find_pivots(closes)
    signals = []
    
    latest_price = closes[-1]
    ts = valid_candles[-1]["timestamp"]
    
    # ─── DOUBLE BOTTOM DETECTION ───
    if len(pivots["lows"]) >= 2:
        l1_idx = pivots["lows"][-1]
        l2_idx = pivots["lows"][-2]
        l1_val = closes[l1_idx]
        l2_val = closes[l2_idx]
        
        # Check if they are at similar levels (within 1.5%)
        diff = abs(l1_val - l2_val) / max(l1_val, l2_val)
        if diff < 0.015:
            # Check if there was a "hump" between them
            hump_max = max(closes[l2_idx:l1_idx])
            if hump_max > l1_val * 1.03: # At least 3% higher
                signals.append({
                    "name": "Double Bottom",
                    "direction": "BUY",
                    "strength": 4,
                    "reason": f"Two touches at approx {l1_val:.2f} — strong support reversal",
                    "timestamp": ts,
                    "price": latest_price
                })

    # ─── DOUBLE TOP DETECTION ───
    if len(pivots["highs"]) >= 2:
        h1_idx = pivots["highs"][-1]
        h2_idx = pivots["highs"][-2]
        h1_val = closes[h1_idx]
        h2_val = closes[h2_idx]
        
        diff = abs(h1_val - h2_val) / max(h1_val, h2_val)
        if diff < 0.015:
            hump_min = min(closes[h2_idx:h1_idx])
            if hump_min < h1_val * 0.97: # At least 3% lower
                signals.append({
                    "name": "Double Top",
                    "direction": "SELL",
                    "strength": 4,
                    "reason": f"Two failed attempts at {h1_val:.2f} — strong resistance reversal",
                    "timestamp": ts,
                    "price": latest_price
                })

    # ─── HORIZONTAL BREAKOUTS ───
    if len(pivots["highs"]) >= 3:
        recent_highs = [closes[i] for i in pivots["highs"][-5:]]
        resistance = max(recent_highs)
        if latest_price > resistance and closes[-2] <= resistance:
            signals.append({
                "name": "Resistance Breakout",
                "direction": "BUY",
                "strength": 5,
                "reason": f"Price broke above horizontal resistance at {resistance:.2f}",
                "timestamp": ts,
                "price": latest_price
            })

    if len(pivots["lows"]) >= 3:
        recent_lows = [closes[i] for i in pivots["lows"][-5:]]
        support = min(recent_lows)
        if latest_price < support and closes[-2] >= support:
            signals.append({
                "name": "Support Breakdown",
                "direction": "SELL",
                "strength": 5,
                "reason": f"Price broke below horizontal support at {support:.2f}",
                "timestamp": ts,
                "price": latest_price
            })

    return signals
