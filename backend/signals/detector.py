"""
Signal Detection Engine — Analyzes patterns and generates BUY/EXIT signals.
Integrates Chart Patterns, Candlestick Patterns, and Technical Indicators.
"""

from datetime import datetime
from backend.signals.pattern_detector import detect_classical_patterns
from backend.signals.candlestick_patterns import detect_candlestick_patterns
from backend.signals.probability_engine import compute_probability
from backend.ml.predictor import predictor

def detect_signals(candles: list[dict], indicators: dict, interval: str = "1h") -> list[dict]:
    """
    Analyze candle data + indicators to detect signals.
    Enriches signals with historical probabilities.
    """
    if not candles or not indicators:
        return []

    signals = []
    
    # 1. ─── CHART PATTERNS ───
    signals.extend(detect_classical_patterns(candles))
    
    # 2. ─── CANDLESTICK PATTERNS ───
    signals.extend(detect_candlestick_patterns(candles))
    
    latest = candles[-1]
    price = latest["close"]
    ts = latest["timestamp"]

    # 3. ─── RSI SIGNALS (Historical Prob: ~58% for reversals) ───
    rsi_data = indicators.get("rsi", [])
    if len(rsi_data) >= 2:
        rsi_now = rsi_data[-1]["value"]
        if rsi_now < 30:
            signals.append(_psig("RSI Oversold", "BUY", 4, 62, f"RSI at {rsi_now:.1f}", ts, price))
        elif rsi_now > 70:
            signals.append(_psig("RSI Overbought", "SELL", 4, 62, f"RSI at {rsi_now:.1f}", ts, price))

    # 4. ─── EMA CROSSOVER (Golden/Death Cross Prob: ~64%) ───
    ema9 = indicators.get("ema_9", [])
    ema21 = indicators.get("ema_21", [])
    if len(ema9) >= 2 and len(ema21) >= 2:
        e9_now, e9_prev = ema9[-1]["value"], ema9[-2]["value"]
        e21_now, e21_prev = ema21[-1]["value"], ema21[-2]["value"]
        if e9_prev <= e21_prev and e9_now > e21_now:
            signals.append(_psig("EMA 9/21 Golden Cross", "BUY", 4, 64, "Momentum shift up", ts, price))
        elif e9_prev >= e21_prev and e9_now < e21_now:
            signals.append(_psig("EMA 9/21 Death Cross", "SELL", 4, 64, "Momentum shift down", ts, price))

    # 5. ─── MACD SIGNALS (Prob: ~60%) ───
    macd_line = indicators.get("macd_line", [])
    macd_signal = indicators.get("macd_signal", [])
    if len(macd_line) >= 2 and len(macd_signal) >= 2:
        ml_now, ml_prev = macd_line[-1]["value"], macd_line[-2]["value"]
        ms_now, ms_prev = macd_signal[-1]["value"], macd_signal[-2]["value"]
        if ml_prev <= ms_prev and ml_now > ms_now:
            signals.append(_psig("MACD Bullish Cross", "BUY", 3, 60, "Bullish convergence", ts, price))
        elif ml_prev >= ms_prev and ml_now < ms_now:
            signals.append(_psig("MACD Bearish Cross", "SELL", 3, 60, "Bearish convergence", ts, price))

    # 6. ─── BOLLINGER BANDS (Prob: ~55%) ───
    bb_upper = indicators.get("bb_upper", [])
    bb_lower = indicators.get("bb_lower", [])
    if bb_upper and bb_lower:
        bbu, bbl = bb_upper[-1]["value"], bb_lower[-1]["value"]
        if price <= bbl:
            signals.append(_psig("Bollinger Bounce", "BUY", 3, 56, "Price at support band", ts, price))
        elif price >= bbu:
            signals.append(_psig("Bollinger Rejection", "SELL", 3, 56, "Price at resistance band", ts, price))

    # ─── CONFLUENCE SCORING ───
    buy_signals = [s for s in signals if s["direction"] == "BUY"]
    sell_signals = [s for s in signals if s["direction"] == "SELL"]
    buy_confluence = sum(s.get("strength", 0) for s in buy_signals)
    sell_confluence = sum(s.get("strength", 0) for s in sell_signals)

    for s in signals:
        s["buy_confluence"] = buy_confluence
        s["sell_confluence"] = sell_confluence

    # ─── ML PREDICTION ───
    ml_result = predictor.predict(indicators, candles, interval)
    if ml_result.get("status") == "ok":
        up = ml_result["up_prob"]
        down = ml_result["down_prob"]
        
        # Add to global stats for the frontend
        if signals:
            signals[0]["ml_prob"] = {"up": up, "down": down}
            
        if up > 80:
            signals.append(_psig("AI ML Prediction", "BUY", 5, up,
                f"Machine Learning model predicts {up}% probability of upward move", ts, price))
            buy_confluence += 5
        elif down > 80:
            signals.append(_psig("AI ML Prediction", "SELL", 5, down,
                f"Machine Learning model predicts {down}% probability of downward move", ts, price))
            sell_confluence += 5

    signals.sort(key=lambda x: x.get("probability", 0), reverse=True)
    return signals


def _psig(name, direction, strength, probability, reason, timestamp, price):
    return {
        "name": name,
        "type": "indicator",
        "direction": direction,
        "strength": min(5, max(1, strength)),
        "probability": probability,
        "reason": reason,
        "timestamp": timestamp,
        "price": price,
        "generated_at": datetime.now().isoformat()
    }
