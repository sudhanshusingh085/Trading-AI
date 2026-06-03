"""
Probability Engine
Combines multiple technical signals and patterns into a single weighted probability score.
Calculates the statistical chance of price moving up vs down.

FIX: BUY and SELL signal pools are now scored independently.
     The old approach divided both pools by total_weight, making strong BUY signals
     get suppressed by even weak SELL signals. This fix properly separates the pools,
     then blends them using a directional balance ratio.
"""

from typing import List, Dict

def compute_probability(signals: List[Dict], indicators: Dict, candles: List[Dict]) -> Dict:
    """
    Aggregate all detected signals into a single probability score.
    Returns:
        {
            "up_probability": float (0-100),
            "down_probability": float (0-100),
            "confidence": str (LOW/MEDIUM/HIGH),
            "prediction": str (BULLISH/BEARISH/NEUTRAL),
            "contributing_patterns": list
        }
    """
    if not signals:
        return {
            "up_probability": 50.0,
            "down_probability": 50.0,
            "confidence": "LOW",
            "prediction": "NEUTRAL",
            "contributing_patterns": []
        }

    # Strength multipliers for weighting
    strength_weights = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.2}

    # Category multipliers (chart patterns more reliable than single indicators)
    category_weights = {
        "chart": 1.2,
        "candlestick": 0.8,
        "indicator": 1.0
    }

    # ── Separate BUY and SELL pools ──────────────────────────────────────────
    buy_weighted_prob = 0.0
    buy_total_weight  = 0.0
    sell_weighted_prob = 0.0
    sell_total_weight  = 0.0
    contributing = []

    for s in signals:
        prob     = s.get("probability", 50.0)
        strength = s.get("strength", 3)
        cat      = s.get("type", "indicator")
        weight   = strength_weights.get(strength, 1.0) * category_weights.get(cat, 1.0)

        if s["direction"] == "BUY":
            buy_weighted_prob += prob * weight
            buy_total_weight  += weight
            contributing.append(s["name"])
        elif s["direction"] == "SELL":
            sell_weighted_prob += prob * weight
            sell_total_weight  += weight
            contributing.append(s["name"])

    # Weighted average probability within each pool (independent of opposing signals)
    avg_buy_prob  = (buy_weighted_prob  / buy_total_weight)  if buy_total_weight  > 0 else 50.0
    avg_sell_prob = (sell_weighted_prob / sell_total_weight) if sell_total_weight > 0 else 50.0

    # ── Directional balance: how much weight is on each side ─────────────────
    total_weight = buy_total_weight + sell_total_weight
    buy_share    = buy_total_weight  / total_weight if total_weight > 0 else 0.5
    sell_share   = sell_total_weight / total_weight if total_weight > 0 else 0.5

    # Final up_probability:
    #   - Fully-BUY market  → avg_buy_prob
    #   - Fully-SELL market → 100 - avg_sell_prob (inverted, so high sell prob → low up_prob)
    #   - Mixed             → weighted blend of both
    up_prob = (buy_share * avg_buy_prob) + (sell_share * (100.0 - avg_sell_prob))

    # ── Apply long-term trend confirmation (EMA 200) ─────────────────────────
    ema200 = indicators.get("ema_200", [])
    if ema200 and len(candles) > 0:
        last_price = candles[-1]["close"]
        last_ema   = ema200[-1]["value"]
        if last_price > last_ema:   # Price above 200 EMA → uptrend confirmation
            up_prob += 4
        else:                       # Price below 200 EMA → downtrend confirmation
            up_prob -= 4

    # ── Clamp ────────────────────────────────────────────────────────────────
    up_prob   = max(10.0, min(90.0, up_prob))
    down_prob = 100.0 - up_prob

    # ── Confidence: based on total signal count ───────────────────────────────
    total_count = len([s for s in signals if s["direction"] in ("BUY", "SELL")])
    if total_count >= 5:
        confidence = "HIGH"
    elif total_count >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Prediction label ──────────────────────────────────────────────────────
    if up_prob > 65:
        prediction = "BULLISH"
    elif up_prob > 55:
        prediction = "SLIGHTLY BULLISH"
    elif up_prob < 35:
        prediction = "BEARISH"
    elif up_prob < 45:
        prediction = "SLIGHTLY BEARISH"
    else:
        prediction = "NEUTRAL"

    return {
        "up_probability":        round(up_prob, 1),
        "down_probability":      round(down_prob, 1),
        "confidence":            confidence,
        "prediction":            prediction,
        "contributing_patterns": list(set(contributing))[:5]
    }
