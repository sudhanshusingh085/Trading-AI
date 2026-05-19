"""
Probability Engine
Combines multiple technical signals and patterns into a single weighted probability score.
Calculates the statistical chance of price moving up vs down.
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

    buy_weighted_sum = 0
    sell_weighted_sum = 0
    total_weight = 0
    
    # Strength multipliers for weighting
    # strength maps {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.2}
    strength_weights = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.2}
    
    # Category multipliers (optional, giving more weight to Chart patterns)
    category_weights = {
        "chart": 1.2,
        "candlestick": 0.8,
        "indicator": 1.0
    }

    contributing = []

    for s in signals:
        prob = s.get("probability", 50.0)
        strength = s.get("strength", 3)
        cat = s.get("type", "indicator")
        
        # Calculate individual weight
        weight = strength_weights.get(strength, 1.0) * category_weights.get(cat, 1.0)
        
        if s["direction"] == "BUY":
            buy_weighted_sum += prob * weight
            total_weight += weight
            contributing.append(s["name"])
        elif s["direction"] == "SELL":
            sell_weighted_sum += prob * weight
            total_weight += weight
            contributing.append(s["name"])
            
    if total_weight == 0:
        return {"up_probability": 50.0, "down_probability": 50.0, "confidence": "LOW", "prediction": "NEUTRAL", "contributing_patterns": []}

    # Normalize to 0-100
    # If all signals are BUY, up_prob will be high. 
    # We use a base-50 normalization
    up_score = (buy_weighted_sum / total_weight) if total_weight > 0 else 50.0
    down_score = (sell_weighted_sum / total_weight) if total_weight > 0 else 50.0
    
    # Final probability calculation
    # If we have both BUY and SELL signals, they balance out
    # We take the difference weighted by signal frequency
    buy_count = len([s for s in signals if s["direction"] == "BUY"])
    sell_count = len([s for s in signals if s["direction"] == "SELL"])
    total_count = buy_count + sell_count
    
    bias = (buy_count - sell_count) / total_count if total_count > 0 else 0
    
    # Resulting probability centered at 50%
    # If bias is 1 (all BUY), up_prob approaches up_score
    # If bias is -1 (all SELL), up_prob approaches (100 - down_score)
    if bias >= 0:
        up_prob = 50 + (up_score - 50) * bias
    else:
        up_prob = 50 + (50 - down_score) * abs(bias)
        
    # Apply trend confirmation boost (EMA 200)
    ema200 = indicators.get("ema_200", [])
    if ema200 and len(candles) > 0:
        last_price = candles[-1]["close"]
        last_ema = ema200[-1]["value"]
        if last_price > last_ema: # Uptrend
            up_prob += 5 # 5% boost for being above long-term trend
        else:
            up_prob -= 5 # 5% penalty for being below
            
    # Clamp results
    up_prob = max(10, min(90, up_prob))
    down_prob = 100 - up_prob
    
    # Determine confidence
    confidence = "LOW"
    if total_count >= 5: confidence = "HIGH"
    elif total_count >= 3: confidence = "MEDIUM"
    
    # Determine prediction
    prediction = "NEUTRAL"
    if up_prob > 65: prediction = "BULLISH"
    elif up_prob > 55: prediction = "SLIGHTLY BULLISH"
    elif up_prob < 35: prediction = "BEARISH"
    elif up_prob < 45: prediction = "SLIGHTLY BEARISH"

    return {
        "up_probability": round(up_prob, 1),
        "down_probability": round(down_prob, 1),
        "confidence": confidence,
        "prediction": prediction,
        "contributing_patterns": list(set(contributing))[:5]
    }
