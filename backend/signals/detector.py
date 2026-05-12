"""
Signal Detection Engine — Analyzes patterns and generates BUY/EXIT signals.
Scans historical data for statistical edges using multiple indicator confluence.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from backend.signals.pattern_detector import detect_classical_patterns


def detect_signals(candles: list[dict], indicators: dict) -> list[dict]:
    """
    Analyze candle data + indicators to detect BUY/SELL signals.
    Returns list of signal dicts sorted by strength (strongest first).
    """
    if not candles or not indicators:
        return []

    signals = []
    
    # ─── CLASSICAL CHART PATTERNS ───
    signals.extend(detect_classical_patterns(candles))
    
    latest = candles[-1]
    prev = candles[-2] if len(candles) > 1 else latest
    price = latest["close"]
    ts = latest["timestamp"]

    # ─── RSI SIGNALS ───
    rsi_data = indicators.get("rsi", [])
    if len(rsi_data) >= 2:
        rsi_now = rsi_data[-1]["value"]
        rsi_prev = rsi_data[-2]["value"]
        if rsi_now < 30:
            signals.append(_sig("RSI Oversold", "BUY", min(5, int((30 - rsi_now) / 5) + 2),
                f"RSI at {rsi_now:.1f} — deeply oversold, reversal likely", ts, price))
        elif rsi_now > 70:
            signals.append(_sig("RSI Overbought", "SELL", min(5, int((rsi_now - 70) / 5) + 2),
                f"RSI at {rsi_now:.1f} — overbought, consider exit", ts, price))
        if rsi_prev < 30 and rsi_now >= 30:
            signals.append(_sig("RSI Cross Above 30", "BUY", 4,
                f"RSI recovered from oversold ({rsi_prev:.1f} → {rsi_now:.1f})", ts, price))
        if rsi_prev > 70 and rsi_now <= 70:
            signals.append(_sig("RSI Cross Below 70", "SELL", 4,
                f"RSI dropped from overbought ({rsi_prev:.1f} → {rsi_now:.1f})", ts, price))

    # ─── EMA CROSSOVER SIGNALS ───
    ema9 = indicators.get("ema_9", [])
    ema21 = indicators.get("ema_21", [])
    if len(ema9) >= 2 and len(ema21) >= 2:
        e9_now, e9_prev = ema9[-1]["value"], ema9[-2]["value"]
        e21_now, e21_prev = ema21[-1]["value"], ema21[-2]["value"]
        if e9_prev <= e21_prev and e9_now > e21_now:
            signals.append(_sig("EMA 9/21 Golden Cross", "BUY", 4,
                "Fast EMA crossed above slow — bullish momentum shift", ts, price))
        elif e9_prev >= e21_prev and e9_now < e21_now:
            signals.append(_sig("EMA 9/21 Death Cross", "SELL", 4,
                "Fast EMA crossed below slow — bearish momentum shift", ts, price))

    # ─── EMA 50/200 (Swing) ───
    ema50 = indicators.get("ema_50", [])
    ema200 = indicators.get("ema_200", [])
    if len(ema50) >= 2 and len(ema200) >= 2:
        e50_now, e50_prev = ema50[-1]["value"], ema50[-2]["value"]
        e200_now, e200_prev = ema200[-1]["value"], ema200[-2]["value"]
        if e50_prev <= e200_prev and e50_now > e200_now:
            signals.append(_sig("EMA 50/200 Golden Cross", "BUY", 5,
                "Major bullish trend confirmed — strong swing buy", ts, price))
        elif e50_prev >= e200_prev and e50_now < e200_now:
            signals.append(_sig("EMA 50/200 Death Cross", "SELL", 5,
                "Major bearish trend confirmed — strong swing exit", ts, price))

    # ─── MACD SIGNALS ───
    macd_line = indicators.get("macd_line", [])
    macd_signal = indicators.get("macd_signal", [])
    macd_hist = indicators.get("macd_histogram", [])
    if len(macd_line) >= 2 and len(macd_signal) >= 2:
        ml_now, ml_prev = macd_line[-1]["value"], macd_line[-2]["value"]
        ms_now, ms_prev = macd_signal[-1]["value"], macd_signal[-2]["value"]
        if ml_prev <= ms_prev and ml_now > ms_now:
            signals.append(_sig("MACD Bullish Cross", "BUY", 3,
                "MACD line crossed above signal — momentum turning bullish", ts, price))
        elif ml_prev >= ms_prev and ml_now < ms_now:
            signals.append(_sig("MACD Bearish Cross", "SELL", 3,
                "MACD line crossed below signal — momentum turning bearish", ts, price))
    if len(macd_hist) >= 3:
        h = [x["value"] for x in macd_hist[-3:]]
        if h[0] < h[1] < h[2] and h[2] > 0:
            signals.append(_sig("MACD Histogram Rising", "BUY", 2,
                "Increasing bullish momentum", ts, price))
        elif h[0] > h[1] > h[2] and h[2] < 0:
            signals.append(_sig("MACD Histogram Falling", "SELL", 2,
                "Increasing bearish momentum", ts, price))

    # ─── BOLLINGER BAND SIGNALS ───
    bb_upper = indicators.get("bb_upper", [])
    bb_lower = indicators.get("bb_lower", [])
    bb_mid = indicators.get("bb_mid", [])
    if bb_upper and bb_lower:
        bbu = bb_upper[-1]["value"]
        bbl = bb_lower[-1]["value"]
        if price <= bbl:
            signals.append(_sig("Bollinger Lower Touch", "BUY", 3,
                f"Price ({price:.2f}) at lower band ({bbl:.2f}) — potential bounce", ts, price))
        elif price >= bbu:
            signals.append(_sig("Bollinger Upper Touch", "SELL", 3,
                f"Price ({price:.2f}) at upper band ({bbu:.2f}) — potential reversal", ts, price))
        # Squeeze detection
        bb_bw = indicators.get("bb_bandwidth", [])
        if len(bb_bw) >= 20:
            recent_bw = [x["value"] for x in bb_bw[-20:]]
            if bb_bw[-1]["value"] == min(recent_bw):
                signals.append(_sig("Bollinger Squeeze", "WATCH", 3,
                    "Bands at tightest in 20 periods — breakout imminent", ts, price))

    # ─── VOLUME SIGNALS ───
    vol_ratio = indicators.get("volume_ratio", [])
    if vol_ratio:
        vr = vol_ratio[-1]["value"]
        if vr > 2.0:
            direction = "BUY" if price > prev["close"] else "SELL"
            signals.append(_sig("Volume Spike", direction, 3,
                f"Volume {vr:.1f}x above average — institutional interest", ts, price))

    # ─── SUPERTREND SIGNALS ───
    st_dir = indicators.get("supertrend_direction", [])
    if len(st_dir) >= 2:
        sd_now = st_dir[-1]["value"]
        sd_prev = st_dir[-2]["value"]
        if sd_prev != sd_now:
            if sd_now == 1:
                signals.append(_sig("Supertrend Bullish Flip", "BUY", 4,
                    "Supertrend flipped bullish — trend reversal up", ts, price))
            else:
                signals.append(_sig("Supertrend Bearish Flip", "SELL", 4,
                    "Supertrend flipped bearish — trend reversal down", ts, price))

    # ─── PRICE ACTION ───
    if len(candles) >= 20:
        highs = [c["high"] for c in candles[-20:]]
        lows = [c["low"] for c in candles[-20:]]
        if price >= max(highs):
            signals.append(_sig("20-Period Breakout High", "BUY", 4,
                f"Price broke above 20-period high — breakout", ts, price))
        if price <= min(lows):
            signals.append(_sig("20-Period Breakout Low", "SELL", 4,
                f"Price broke below 20-period low — breakdown", ts, price))

    # ─── CONFLUENCE SCORING ───
    buy_signals = [s for s in signals if s["direction"] == "BUY"]
    sell_signals = [s for s in signals if s["direction"] == "SELL"]
    buy_confluence = sum(s["strength"] for s in buy_signals)
    sell_confluence = sum(s["strength"] for s in sell_signals)

    for s in signals:
        s["buy_confluence"] = buy_confluence
        s["sell_confluence"] = sell_confluence

    signals.sort(key=lambda x: x["strength"], reverse=True)
    return signals


def _sig(name, direction, strength, reason, timestamp, price):
    return {
        "name": name,
        "direction": direction,
        "strength": min(5, max(1, strength)),
        "reason": reason,
        "timestamp": timestamp,
        "price": price,
        "generated_at": datetime.now().isoformat()
    }
