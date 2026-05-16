"""
Candlestick Pattern Detection Engine
Detects single and multi-candle patterns with historical success probabilities.
Source: Bulkowski's Encyclopedia of Candlestick Charts (simplified).
"""

from typing import List, Dict

def detect_candlestick_patterns(candles: List[Dict]) -> List[Dict]:
    """
    Detect candlestick patterns from the latest OHLCV data.
    Returns a list of signal dicts.
    """
    if len(candles) < 5:
        return []

    signals = []
    
    # Extract latest 3 candles for analysis
    c1 = candles[-1] # current
    c2 = candles[-2] # previous
    c3 = candles[-3] # 2nd previous
    
    # Basic properties
    def is_bull(c): return c["close"] > c["open"]
    def is_bear(c): return c["close"] < c["open"]
    def body_size(c): return abs(c["close"] - c["open"])
    def total_size(c): return c["high"] - c["low"]
    
    ts = c1["timestamp"]
    price = c1["close"]

    # ─── BULLISH ENGULFING (Prob: 63%) ───
    if is_bear(c2) and is_bull(c1) and \
       c1["open"] <= c2["close"] and c1["close"] > c2["open"]:
        signals.append({
            "name": "Bullish Engulfing",
            "type": "candlestick",
            "direction": "BUY",
            "strength": 4,
            "probability": 63,
            "reason": "Strong bullish candle fully engulfs the body of the previous bearish candle",
            "timestamp": ts, "price": price
        })

    # ─── BEARISH ENGULFING (Prob: 79%) ───
    if is_bull(c2) and is_bear(c1) and \
       c1["open"] >= c2["close"] and c1["close"] < c2["open"]:
        signals.append({
            "name": "Bearish Engulfing",
            "type": "candlestick",
            "direction": "SELL",
            "strength": 4,
            "probability": 79,
            "reason": "Strong bearish candle fully engulfs the body of the previous bullish candle",
            "timestamp": ts, "price": price
        })

    # ─── HAMMER (Prob: 60%) ───
    # Small body, long lower wick, little to no upper wick
    lower_wick = min(c1["open"], c1["close"]) - c1["low"]
    upper_wick = c1["high"] - max(c1["open"], c1["close"])
    if lower_wick > 2 * body_size(c1) and upper_wick < 0.1 * total_size(c1):
        signals.append({
            "name": "Hammer",
            "type": "candlestick",
            "direction": "BUY",
            "strength": 3,
            "probability": 60,
            "reason": "Bullish pin bar indicating rejection of lower prices",
            "timestamp": ts, "price": price
        })

    # ─── HANGING MAN (Prob: 59%) ───
    # Same shape as hammer but after an uptrend
    if lower_wick > 2 * body_size(c1) and upper_wick < 0.1 * total_size(c1) and c1["close"] > candles[-5]["close"]:
        signals.append({
            "name": "Hanging Man",
            "type": "candlestick",
            "direction": "SELL",
            "strength": 3,
            "probability": 59,
            "reason": "Bearish reversal pin bar occurring after an uptrend",
            "timestamp": ts, "price": price
        })

    # ─── MORNING STAR (Prob: 78%) ───
    if is_bear(c3) and body_size(c2) < 0.3 * body_size(c3) and is_bull(c1) and \
       c1["close"] > (c3["open"] + c3["close"]) / 2:
        signals.append({
            "name": "Morning Star",
            "type": "candlestick",
            "direction": "BUY",
            "strength": 5,
            "probability": 78,
            "reason": "3-candle bottom reversal pattern indicating bullish recovery",
            "timestamp": ts, "price": price
        })

    # ─── EVENING STAR (Prob: 72%) ───
    if is_bull(c3) and body_size(c2) < 0.3 * body_size(c3) and is_bear(c1) and \
       c1["close"] < (c3["open"] + c3["close"]) / 2:
        signals.append({
            "name": "Evening Star",
            "type": "candlestick",
            "direction": "SELL",
            "strength": 5,
            "probability": 72,
            "reason": "3-candle top reversal pattern indicating bearish turn",
            "timestamp": ts, "price": price
        })

    # ─── THREE WHITE SOLDIERS (Prob: 82%) ───
    if is_bull(c1) and is_bull(c2) and is_bull(c3) and \
       c1["close"] > c2["close"] > c3["close"] and \
       body_size(c1) > 0.5 * body_size(c2):
        signals.append({
            "name": "Three White Soldiers",
            "type": "candlestick",
            "direction": "BUY",
            "strength": 5,
            "probability": 82,
            "reason": "Three strong consecutive bullish candles indicating strong uptrend",
            "timestamp": ts, "price": price
        })

    # ─── THREE BLACK CROWS (Prob: 78%) ───
    if is_bear(c1) and is_bear(c2) and is_bear(c3) and \
       c1["close"] < c2["close"] < c3["close"] and \
       body_size(c1) > 0.5 * body_size(c2):
        signals.append({
            "name": "Three Black Crows",
            "type": "candlestick",
            "direction": "SELL",
            "strength": 5,
            "probability": 78,
            "reason": "Three strong consecutive bearish candles indicating strong downtrend",
            "timestamp": ts, "price": price
        })

    # ─── DOJI (Prob: 51%) ───
    if body_size(c1) < 0.1 * total_size(c1):
        signals.append({
            "name": "Doji",
            "type": "candlestick",
            "direction": "WATCH",
            "strength": 1,
            "probability": 51,
            "reason": "Market indecision, potential turning point ahead",
            "timestamp": ts, "price": price
        })

    # ─── PIERCING LINE (Prob: 64%) ───
    if is_bear(c2) and is_bull(c1) and \
       c1["open"] < c2["low"] and c1["close"] > (c2["open"] + c2["close"]) / 2:
        signals.append({
            "name": "Piercing Line",
            "type": "candlestick",
            "direction": "BUY",
            "strength": 4,
            "probability": 64,
            "reason": "Bullish reversal where current candle pierces 50% of the previous bearish candle",
            "timestamp": ts, "price": price
        })

    # ─── DARK CLOUD COVER (Prob: 60%) ───
    if is_bull(c2) and is_bear(c1) and \
       c1["open"] > c2["high"] and c1["close"] < (c2["open"] + c2["close"]) / 2:
        signals.append({
            "name": "Dark Cloud Cover",
            "type": "candlestick",
            "direction": "SELL",
            "strength": 4,
            "probability": 60,
            "reason": "Bearish reversal where current candle closes below 50% of the previous bullish candle",
            "timestamp": ts, "price": price
        })

    return signals
