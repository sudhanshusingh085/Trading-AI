"""
Technical Indicator Calculator
Pure numpy/pandas implementation — no pandas-ta dependency (Python 3.14 compatible).
Computes 15+ indicators on OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_all_indicators(candles: list[dict], config: dict = None) -> dict:
    """Compute all technical indicators on OHLCV candle data."""
    if not candles or len(candles) < 2:
        return {"error": "Not enough candle data"}

    df = pd.DataFrame(candles)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cfg = {
        "ema_fast": 9, "ema_mid": 21, "ema_slow": 50, "ema_trend": 200,
        "sma_20": 20, "rsi_period": 14, "macd_fast": 12, "macd_slow": 26,
        "macd_signal": 9, "bb_period": 20, "bb_std": 2.0, "atr_period": 14,
        "stoch_k": 14, "stoch_d": 3, "volume_sma": 20,
    }
    if config:
        cfg.update(config)

    indicators = {}
    timestamps = df["timestamp"].tolist()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ─── TREND: EMA ───
    for label, period in [("ema_9", cfg["ema_fast"]), ("ema_21", cfg["ema_mid"]),
                           ("ema_50", cfg["ema_slow"]), ("ema_200", cfg["ema_trend"])]:
        if len(close) >= period:
            indicators[label] = _series_to_list(timestamps, _ema(close, period))

    # SMA 20
    if len(close) >= cfg["sma_20"]:
        indicators["sma_20"] = _series_to_list(timestamps, close.rolling(cfg["sma_20"]).mean())

    # VWAP approximation
    try:
        typical = (high + low + close) / 3
        vwap = (volume * typical).cumsum() / volume.cumsum()
        indicators["vwap"] = _series_to_list(timestamps, vwap)
    except Exception:
        pass

    # Supertrend
    try:
        st, st_dir = _supertrend(high, low, close, 10, 3.0)
        indicators["supertrend"] = _series_to_list(timestamps, st)
        indicators["supertrend_direction"] = _series_to_list(timestamps, st_dir)
    except Exception:
        pass

    # ─── MOMENTUM: RSI ───
    rsi = _rsi(close, cfg["rsi_period"])
    if rsi is not None:
        indicators["rsi"] = _series_to_list(timestamps, rsi)

    # MACD
    if len(close) >= cfg["macd_slow"]:
        ema_f = _ema(close, cfg["macd_fast"])
        ema_s = _ema(close, cfg["macd_slow"])
        macd_line = ema_f - ema_s
        macd_signal = _ema(macd_line, cfg["macd_signal"])
        macd_hist = macd_line - macd_signal
        indicators["macd_line"] = _series_to_list(timestamps, macd_line)
        indicators["macd_signal"] = _series_to_list(timestamps, macd_signal)
        indicators["macd_histogram"] = _series_to_list(timestamps, macd_hist)

    # Stochastic
    if len(close) >= cfg["stoch_k"]:
        low_min = low.rolling(cfg["stoch_k"]).min()
        high_max = high.rolling(cfg["stoch_k"]).max()
        k = ((close - low_min) / (high_max - low_min)) * 100
        d = k.rolling(cfg["stoch_d"]).mean()
        indicators["stoch_k"] = _series_to_list(timestamps, k)
        indicators["stoch_d"] = _series_to_list(timestamps, d)

    # ADX
    try:
        adx = _adx(high, low, close, 14)
        if adx is not None:
            indicators["adx"] = _series_to_list(timestamps, adx)
    except Exception:
        pass

    # ─── VOLATILITY: Bollinger Bands ───
    if len(close) >= cfg["bb_period"]:
        sma = close.rolling(cfg["bb_period"]).mean()
        std = close.rolling(cfg["bb_period"]).std()
        indicators["bb_upper"] = _series_to_list(timestamps, sma + cfg["bb_std"] * std)
        indicators["bb_mid"] = _series_to_list(timestamps, sma)
        indicators["bb_lower"] = _series_to_list(timestamps, sma - cfg["bb_std"] * std)
        bw = ((sma + cfg["bb_std"] * std) - (sma - cfg["bb_std"] * std)) / sma * 100
        indicators["bb_bandwidth"] = _series_to_list(timestamps, bw)

    # ATR
    if len(close) >= cfg["atr_period"] + 1:
        tr = _true_range(high, low, close)
        atr = tr.rolling(cfg["atr_period"]).mean()
        indicators["atr"] = _series_to_list(timestamps, atr)

    # ─── VOLUME ───
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    indicators["obv"] = _series_to_list(timestamps, obv)

    vol_sma = volume.rolling(cfg["volume_sma"]).mean()
    indicators["volume_sma"] = _series_to_list(timestamps, vol_sma)

    vol_ratio = volume / vol_sma
    indicators["volume_ratio"] = _series_to_list(timestamps, vol_ratio)

    return indicators


# ─── Helper functions ───

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = _true_range(high, low, close)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()

def _supertrend(high, low, close, period=10, multiplier=3.0):
    tr = _true_range(high, low, close)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)
    for i in range(period, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i - 1]:
                lower.iloc[i] = lower.iloc[i - 1]
            if direction.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i - 1]:
                upper.iloc[i] = upper.iloc[i - 1]
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return st, direction

def _series_to_list(timestamps, series):
    result = []
    for ts, val in zip(timestamps, series):
        if pd.notna(val) and not np.isinf(val):
            result.append({"time": ts, "value": round(float(val), 6)})
    return result

def get_available_indicators():
    return [
        {"name": "ema_9", "category": "trend", "description": "EMA 9", "overlay": True},
        {"name": "ema_21", "category": "trend", "description": "EMA 21", "overlay": True},
        {"name": "ema_50", "category": "trend", "description": "EMA 50", "overlay": True},
        {"name": "ema_200", "category": "trend", "description": "EMA 200", "overlay": True},
        {"name": "sma_20", "category": "trend", "description": "SMA 20", "overlay": True},
        {"name": "vwap", "category": "trend", "description": "VWAP", "overlay": True},
        {"name": "supertrend", "category": "trend", "description": "Supertrend", "overlay": True},
        {"name": "rsi", "category": "momentum", "description": "RSI 14", "overlay": False},
        {"name": "macd_line", "category": "momentum", "description": "MACD Line", "overlay": False},
        {"name": "macd_signal", "category": "momentum", "description": "MACD Signal", "overlay": False},
        {"name": "macd_histogram", "category": "momentum", "description": "MACD Histogram", "overlay": False},
        {"name": "stoch_k", "category": "momentum", "description": "Stochastic %K", "overlay": False},
        {"name": "stoch_d", "category": "momentum", "description": "Stochastic %D", "overlay": False},
        {"name": "adx", "category": "momentum", "description": "ADX", "overlay": False},
        {"name": "bb_upper", "category": "volatility", "description": "Bollinger Upper", "overlay": True},
        {"name": "bb_mid", "category": "volatility", "description": "Bollinger Mid", "overlay": True},
        {"name": "bb_lower", "category": "volatility", "description": "Bollinger Lower", "overlay": True},
        {"name": "atr", "category": "volatility", "description": "ATR 14", "overlay": False},
        {"name": "obv", "category": "volume", "description": "OBV", "overlay": False},
        {"name": "volume_sma", "category": "volume", "description": "Volume SMA", "overlay": False},
        {"name": "volume_ratio", "category": "volume", "description": "Volume Ratio", "overlay": False},
    ]
