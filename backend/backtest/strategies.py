"""
Pre-built Trading Strategies for backtesting.
Each strategy is a function: (candles, indicators, current_index) -> "BUY" | "SELL" | None
"""


def ema_crossover_9_21(candles, indicators, i):
    """EMA 9/21 Crossover — Good for intraday/short-term swing."""
    ema9 = indicators.get("ema_9", [])
    ema21 = indicators.get("ema_21", [])
    if i < 2 or i >= len(ema9) or i >= len(ema21):
        return None
    if ema9[i-1]["value"] <= ema21[i-1]["value"] and ema9[i]["value"] > ema21[i]["value"]:
        return "BUY"
    if ema9[i-1]["value"] >= ema21[i-1]["value"] and ema9[i]["value"] < ema21[i]["value"]:
        return "SELL"
    return None


def ema_crossover_50_200(candles, indicators, i):
    """EMA 50/200 Crossover — Classic swing/position trading."""
    ema50 = indicators.get("ema_50", [])
    ema200 = indicators.get("ema_200", [])
    if i < 2 or i >= len(ema50) or i >= len(ema200):
        return None
    if ema50[i-1]["value"] <= ema200[i-1]["value"] and ema50[i]["value"] > ema200[i]["value"]:
        return "BUY"
    if ema50[i-1]["value"] >= ema200[i-1]["value"] and ema50[i]["value"] < ema200[i]["value"]:
        return "SELL"
    return None


def rsi_mean_reversion(candles, indicators, i):
    """RSI Mean Reversion — Buy oversold, sell overbought."""
    rsi = indicators.get("rsi", [])
    if i >= len(rsi):
        return None
    val = rsi[i]["value"]
    if val < 30:
        return "BUY"
    if val > 70:
        return "SELL"
    return None


def bollinger_bounce(candles, indicators, i):
    """Bollinger Band Bounce — Buy at lower band, sell at upper."""
    bb_lower = indicators.get("bb_lower", [])
    bb_upper = indicators.get("bb_upper", [])
    if i >= len(bb_lower) or i >= len(bb_upper):
        return None
    price = candles[i]["close"]
    if price <= bb_lower[i]["value"]:
        return "BUY"
    if price >= bb_upper[i]["value"]:
        return "SELL"
    return None


def macd_signal_cross(candles, indicators, i):
    """MACD Signal Line Cross."""
    ml = indicators.get("macd_line", [])
    ms = indicators.get("macd_signal", [])
    if i < 2 or i >= len(ml) or i >= len(ms):
        return None
    if ml[i-1]["value"] <= ms[i-1]["value"] and ml[i]["value"] > ms[i]["value"]:
        return "BUY"
    if ml[i-1]["value"] >= ms[i-1]["value"] and ml[i]["value"] < ms[i]["value"]:
        return "SELL"
    return None


def supertrend_follow(candles, indicators, i):
    """Supertrend Follower — Follow the supertrend direction."""
    st_dir = indicators.get("supertrend_direction", [])
    if i < 2 or i >= len(st_dir):
        return None
    if st_dir[i-1]["value"] != st_dir[i]["value"]:
        return "BUY" if st_dir[i]["value"] == 1 else "SELL"
    return None


def breakout_20(candles, indicators, i):
    """20-Period Breakout — Buy on new highs with volume confirmation."""
    if i < 20:
        return None
    vol_ratio = indicators.get("volume_ratio", [])
    highs = [c["high"] for c in candles[i-20:i]]
    lows = [c["low"] for c in candles[i-20:i]]
    price = candles[i]["close"]
    has_volume = i < len(vol_ratio) and vol_ratio[i]["value"] > 1.5

    if price > max(highs) and has_volume:
        return "BUY"
    if price < min(lows):
        return "SELL"
    return None


# Strategy registry
STRATEGIES = {
    "ema_9_21": {"fn": ema_crossover_9_21, "name": "EMA 9/21 Crossover", "type": "intraday",
                  "description": "Buy when EMA 9 crosses above EMA 21, sell on reverse. Best for short-term trades."},
    "ema_50_200": {"fn": ema_crossover_50_200, "name": "EMA 50/200 Golden Cross", "type": "swing",
                    "description": "Classic golden/death cross. Buy when EMA 50 crosses above EMA 200."},
    "rsi_reversion": {"fn": rsi_mean_reversion, "name": "RSI Mean Reversion", "type": "both",
                       "description": "Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)."},
    "bollinger_bounce": {"fn": bollinger_bounce, "name": "Bollinger Band Bounce", "type": "both",
                          "description": "Buy at lower Bollinger Band, sell at upper band."},
    "macd_cross": {"fn": macd_signal_cross, "name": "MACD Signal Cross", "type": "both",
                    "description": "Buy/sell on MACD line crossing signal line."},
    "supertrend": {"fn": supertrend_follow, "name": "Supertrend Follower", "type": "both",
                    "description": "Follow supertrend direction flips."},
    "breakout_20": {"fn": breakout_20, "name": "20-Period Breakout", "type": "swing",
                     "description": "Buy on 20-period high breakout with volume confirmation."},
}
