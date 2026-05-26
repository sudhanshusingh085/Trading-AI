"""
Signal Detection Engine — Analyzes patterns and generates BUY/SELL/EXIT signals.
Integrates Chart Patterns, Candlestick Patterns, and Technical Indicators.

Key changes from original:
- Tiered ML thresholds (65/75/85%) instead of all-or-nothing 80%
- EXIT signal generation for position management
- Signal role metadata (ENTRY vs EXIT) for buy→sell pairing
"""

from datetime import datetime
from backend.signals.pattern_detector import detect_classical_patterns
from backend.signals.candlestick_patterns import detect_candlestick_patterns
from backend.signals.probability_engine import compute_probability
from backend.ml.predictor import predictor

def detect_signals(candles: list[dict], indicators: dict, interval: str = "1h") -> list[dict]:
    """
    Analyze candle data + indicators to detect signals.
    Enriches signals with historical probabilities and signal roles.
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
        rsi_prev = rsi_data[-2]["value"]
        if rsi_now < 30:
            signals.append(_psig("RSI Oversold", "BUY", 4, 62, f"RSI at {rsi_now:.1f}", ts, price, role="ENTRY"))
        elif rsi_now > 70:
            signals.append(_psig("RSI Overbought", "SELL", 4, 62, f"RSI at {rsi_now:.1f}", ts, price, role="ENTRY"))
        
        # EXIT signals: RSI reversing from position direction
        if rsi_prev < 70 and rsi_now >= 70:
            signals.append(_psig("RSI Reaching Overbought", "SELL", 3, 58, f"RSI crossed above 70 ({rsi_now:.1f})", ts, price, role="EXIT"))
        elif rsi_prev > 30 and rsi_now <= 30:
            signals.append(_psig("RSI Reaching Oversold", "BUY", 3, 58, f"RSI crossed below 30 ({rsi_now:.1f})", ts, price, role="EXIT"))

    # 4. ─── EMA CROSSOVER (Golden/Death Cross Prob: ~64%) ───
    ema9 = indicators.get("ema_9", [])
    ema21 = indicators.get("ema_21", [])
    if len(ema9) >= 2 and len(ema21) >= 2:
        e9_now, e9_prev = ema9[-1]["value"], ema9[-2]["value"]
        e21_now, e21_prev = ema21[-1]["value"], ema21[-2]["value"]
        if e9_prev <= e21_prev and e9_now > e21_now:
            signals.append(_psig("EMA 9/21 Golden Cross", "BUY", 4, 64, "Momentum shift up", ts, price, role="ENTRY"))
        elif e9_prev >= e21_prev and e9_now < e21_now:
            signals.append(_psig("EMA 9/21 Death Cross", "SELL", 4, 64, "Momentum shift down", ts, price, role="ENTRY"))
        
        # EXIT: trend weakening (EMA gap narrowing)
        ema_gap_now = abs(e9_now - e21_now) / e21_now * 100
        ema_gap_prev = abs(e9_prev - e21_prev) / e21_prev * 100
        if ema_gap_now < ema_gap_prev * 0.5 and ema_gap_prev > 0.1:
            if e9_now > e21_now:
                signals.append(_psig("EMA Momentum Fading", "SELL", 2, 55, "Bullish EMA gap narrowing rapidly", ts, price, role="EXIT"))
            else:
                signals.append(_psig("EMA Momentum Fading", "BUY", 2, 55, "Bearish EMA gap narrowing rapidly", ts, price, role="EXIT"))

    # 5. ─── MACD SIGNALS (Prob: ~60%) ───
    macd_line = indicators.get("macd_line", [])
    macd_signal = indicators.get("macd_signal", [])
    macd_hist = indicators.get("macd_histogram", [])
    if len(macd_line) >= 2 and len(macd_signal) >= 2:
        ml_now, ml_prev = macd_line[-1]["value"], macd_line[-2]["value"]
        ms_now, ms_prev = macd_signal[-1]["value"], macd_signal[-2]["value"]
        if ml_prev <= ms_prev and ml_now > ms_now:
            signals.append(_psig("MACD Bullish Cross", "BUY", 3, 60, "Bullish convergence", ts, price, role="ENTRY"))
        elif ml_prev >= ms_prev and ml_now < ms_now:
            signals.append(_psig("MACD Bearish Cross", "SELL", 3, 60, "Bearish convergence", ts, price, role="ENTRY"))
    
    # EXIT: MACD histogram declining (momentum loss)
    if len(macd_hist) >= 3:
        h0 = macd_hist[-1]["value"]
        h1 = macd_hist[-2]["value"]
        h2 = macd_hist[-3]["value"]
        # Histogram was positive and declining for 2 bars
        if h2 > h1 > 0 and h1 > h0 and h0 > 0:
            signals.append(_psig("MACD Momentum Weakening", "SELL", 2, 55, "MACD histogram declining while positive", ts, price, role="EXIT"))
        # Histogram was negative and recovering for 2 bars
        elif h2 < h1 < 0 and h1 < h0 and h0 < 0:
            signals.append(_psig("MACD Momentum Recovering", "BUY", 2, 55, "MACD histogram recovering while negative", ts, price, role="EXIT"))

    # 6. ─── BOLLINGER BANDS (Prob: ~55%) ───
    bb_upper = indicators.get("bb_upper", [])
    bb_lower = indicators.get("bb_lower", [])
    if bb_upper and bb_lower:
        bbu, bbl = bb_upper[-1]["value"], bb_lower[-1]["value"]
        if price <= bbl:
            signals.append(_psig("Bollinger Bounce", "BUY", 3, 56, "Price at support band", ts, price, role="ENTRY"))
        elif price >= bbu:
            signals.append(_psig("Bollinger Rejection", "SELL", 3, 56, "Price at resistance band", ts, price, role="ENTRY"))

    # ─── CONFLUENCE SCORING ───
    buy_signals = [s for s in signals if s["direction"] == "BUY"]
    sell_signals = [s for s in signals if s["direction"] == "SELL"]
    buy_confluence = sum(s.get("strength", 0) for s in buy_signals)
    sell_confluence = sum(s.get("strength", 0) for s in sell_signals)

    for s in signals:
        s["buy_confluence"] = buy_confluence
        s["sell_confluence"] = sell_confluence

    # ─── ML PREDICTION (TIERED THRESHOLDS) ───
    ml_result = predictor.predict(indicators, candles, interval)
    if ml_result.get("status") == "ok":
        up = ml_result["up_prob"]
        down = ml_result["down_prob"]
        
        # Add to global stats for the frontend
        if signals:
            signals[0]["ml_prob"] = {"up": up, "down": down}
            
        # Tiered ML signal generation (replaces old 80% all-or-nothing gate)
        if up >= 85:
            signals.append(_psig("AI ML Prediction", "BUY", 5, up,
                f"ML model: {up}% upward probability (very strong)", ts, price, role="ENTRY"))
            buy_confluence += 5
        elif up >= 75:
            signals.append(_psig("AI ML Prediction", "BUY", 4, up,
                f"ML model: {up}% upward probability (strong)", ts, price, role="ENTRY"))
            buy_confluence += 4
        elif up >= 65:
            signals.append(_psig("AI ML Prediction", "BUY", 3, up,
                f"ML model: {up}% upward probability (moderate)", ts, price, role="ENTRY"))
            buy_confluence += 3
        elif down >= 85:
            signals.append(_psig("AI ML Prediction", "SELL", 5, down,
                f"ML model: {down}% downward probability (very strong)", ts, price, role="ENTRY"))
            sell_confluence += 5
        elif down >= 75:
            signals.append(_psig("AI ML Prediction", "SELL", 4, down,
                f"ML model: {down}% downward probability (strong)", ts, price, role="ENTRY"))
            sell_confluence += 4
        elif down >= 65:
            signals.append(_psig("AI ML Prediction", "SELL", 3, down,
                f"ML model: {down}% downward probability (moderate)", ts, price, role="ENTRY"))
            sell_confluence += 3

    signals.sort(key=lambda x: x.get("probability", 0), reverse=True)
    return signals


def _psig(name, direction, strength, probability, reason, timestamp, price, role="ENTRY"):
    return {
        "name": name,
        "type": "indicator",
        "direction": direction,
        "strength": min(5, max(1, strength)),
        "probability": probability,
        "reason": reason,
        "signal_role": role,  # "ENTRY" or "EXIT" — for buy→sell pairing
        "timestamp": timestamp,
        "price": price,
        "generated_at": datetime.now().isoformat()
    }
