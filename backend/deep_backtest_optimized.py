"""
Deep Backtest Optimized — Historical backtest validating the P1 fixes:
  1. Bidirectional LONG/SHORT setups.
  2. Dynamic ATR-based stops (1.5x ATR SL, 3.0x ATR TP -> 2:1 R:R).
  3. Volume confirmation (volume_ratio >= 1.5).
  4. Increased hold threshold (MIN_HOLD = 5 candles).
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
from datetime import datetime
from backend.indicators.calculator import compute_all_indicators
from backend.signals.detector import detect_signals

# ─── CONFIG ───
SYMBOL        = "BTCUSDT"
INTERVAL      = "1h"
LIMIT         = 1000          # ~41 days of 1h candles
INITIAL_CAP   = 10_000.0
POS_SIZE_PCT  = 10.0          # 10% of capital per trade
COMMISSION    = 0.1           # 0.1% per side
LOOKBACK      = 0             
TEST_CANDLES  = 1000          # Run full dataset

MIN_HOLD      = 5             # Increased hold threshold (5 hours / candles)
COOLDOWN      = 3
MIN_CONF      = 6

LONG_EXIT_PATTERNS = {
    "Bearish Engulfing", "Evening Star", "Dark Cloud Cover",
    "Head and Shoulders", "Double Top", "MACD Bearish Cross",
    "EMA 9/21 Death Cross", "Three White Soldiers (Exhaustion)",
    "Descending Triangle", "Bollinger Rejection", "RSI Overbought"
}

SHORT_EXIT_PATTERNS = {
    "Bullish Engulfing", "Morning Star", "Piercing Line",
    "Inverse Head and Shoulders", "Double Bottom", "MACD Bullish Cross",
    "EMA 9/21 Golden Cross", "Three White Soldiers",
    "Ascending Triangle", "Bollinger Breakout", "RSI Oversold"
}


def fetch_candles():
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    data   = requests.get(url, params=params, timeout=10).json()
    return [{
        "timestamp": datetime.fromtimestamp(k[0]/1000).strftime("%Y-%m-%d %H:%M:%S"),
        "open": float(k[1]), "high": float(k[2]),
        "low":  float(k[3]), "close": float(k[4]),
        "volume": float(k[5]), "quote_volume": float(k[7]),
        "trades": int(k[8])
    } for k in data]


def verdict_calc(signals):
    if not signals: return "NEUTRAL"
    act  = [s for s in signals if s.get("direction") in ("BUY","SELL")]
    if len(act) < 2: return "NEUTRAL"
    buy  = sum(s["strength"] for s in act if s["direction"] == "BUY")
    sell = sum(s["strength"] for s in act if s["direction"] == "SELL")
    if abs(buy - sell) < 3: return "NEUTRAL"
    if buy  > sell + 6: return "STRONG BUY"
    if buy  > sell + 3: return "BUY"
    if sell > buy  + 6: return "STRONG SELL"
    if sell > buy  + 3: return "SELL"
    return "NEUTRAL"


def confidence_score(signals, direction):
    aligned   = [s for s in signals if s.get("direction") == direction]
    opposing  = [s for s in signals if s.get("direction") != direction and s.get("direction") in ("BUY","SELL")]
    a_strength = sum(s["strength"] for s in aligned)
    o_strength = sum(s["strength"] for s in opposing)
    total_str  = a_strength + o_strength if (a_strength + o_strength) > 0 else 1
    raw = (a_strength / total_str) * 100
    if len(aligned) >= 3: raw = min(100, raw + 10)
    return round(raw, 1)


def market_regime(indicators):
    adx_data = indicators.get("adx", [])
    if isinstance(adx_data, list) and adx_data:
        adx = adx_data[-1].get("value", 0)
    else:
        adx = float(adx_data) if isinstance(adx_data, (int, float)) else 0
    if adx > 30: return "TRENDING"
    if adx > 18: return "MILD_TREND"
    return "RANGING"


def classify_setup(signals, direction):
    names = {s["name"] for s in signals}
    if "MACD Bullish Cross" in names or "EMA 9/21 Golden Cross" in names:
        return "TREND_CONTINUATION"
    if "Inverse Head and Shoulders" in names or "Double Bottom" in names:
        return "REVERSAL_STRUCTURE"
    if "Breakout" in " ".join(names):
        return "BREAKOUT_RETEST"
    if "RSI Oversold" in names or "RSI Overbought" in names:
        return "MEAN_REVERSION"
    if "Bollinger" in " ".join(names):
        return "VOLATILITY_BREAKOUT"
    return "MOMENTUM"


class OptimizedTracker:
    def __init__(self, name):
        self.name              = name
        self.capital           = INITIAL_CAP
        self.position          = None  # {entry, qty, size, direction, sl, tp, time, held, conf, regime, setup, entry_signals}
        self.trades            = []
        self.state             = "IDLE"
        self.cooldown_rem      = 0
        self.peak_capital      = INITIAL_CAP
        self.max_drawdown      = 0.0
        self.equity_curve      = [INITIAL_CAP]

    def on_candle(self, should_enter_long, should_enter_short, exit_signal, price, ts,
                  buy_conf, sell_conf, signals, indicators):
        if self.state == "COOLDOWN":
            self.cooldown_rem -= 1
            if self.cooldown_rem <= 0:
                self.state = "IDLE"
            self._update_equity(price)
            return None

        # Extract latest ATR
        atr_list = indicators.get("atr", [])
        latest_atr = atr_list[-1]["value"] if isinstance(atr_list, list) and atr_list else price * 0.002

        if self.state == "IN_POSITION":
            self.position["held"] += 1
            direction = self.position["direction"]
            sl = self.position["sl"]
            tp = self.position["tp"]

            # 1. ATR stops checks
            if direction == "LONG":
                if price <= sl:
                    return self._close(sl, ts, "STOP_LOSS", signals, indicators)
                if price >= tp:
                    return self._close(tp, ts, "TAKE_PROFIT", signals, indicators)
            else:  # SHORT
                if price >= sl:
                    return self._close(sl, ts, "STOP_LOSS", signals, indicators)
                if price <= tp:
                    return self._close(tp, ts, "TAKE_PROFIT", signals, indicators)

            # 2. Signal exit check (only after MIN_HOLD)
            if self.position["held"] >= MIN_HOLD and exit_signal:
                return self._close(price, ts, "EXIT_SIGNAL", signals, indicators)

            self._update_equity(price)
            return None

        if self.state == "IDLE":
            size = self.capital * (POS_SIZE_PCT / 100)
            qty  = size / price

            if should_enter_long:
                sl_price = price - (1.5 * latest_atr)
                tp_price = price + (3.0 * latest_atr)
                
                self.capital -= size * (COMMISSION / 100)
                conf = confidence_score(signals, "BUY")
                regime = market_regime(indicators)
                setup  = classify_setup(signals, "BUY")
                self.position = {
                    "entry": price, "qty": qty, "size": size, "direction": "LONG",
                    "sl": sl_price, "tp": tp_price,
                    "time": ts, "held": 0,
                    "conf": conf, "regime": regime, "setup": setup,
                    "entry_signals": [s["name"] for s in signals]
                }
                self.state = "IN_POSITION"
                self._update_equity(price)
                return "ENTER_LONG"
                
            elif should_enter_short:
                sl_price = price + (1.5 * latest_atr)
                tp_price = price - (3.0 * latest_atr)
                
                self.capital -= size * (COMMISSION / 100)
                conf = confidence_score(signals, "SELL")
                regime = market_regime(indicators)
                setup  = classify_setup(signals, "SELL")
                self.position = {
                    "entry": price, "qty": qty, "size": size, "direction": "SHORT",
                    "sl": sl_price, "tp": tp_price,
                    "time": ts, "held": 0,
                    "conf": conf, "regime": regime, "setup": setup,
                    "entry_signals": [s["name"] for s in signals]
                }
                self.state = "IN_POSITION"
                self._update_equity(price)
                return "ENTER_SHORT"

        self._update_equity(price)
        return None

    def _close(self, price, ts, reason, signals, indicators):
        entry_price = self.position["entry"]
        qty = self.position["qty"]
        direction = self.position["direction"]
        
        if direction == "LONG":
            raw_pnl = (price - entry_price) * qty
        else:
            raw_pnl = (entry_price - price) * qty
            
        comm = abs(raw_pnl) * (COMMISSION / 100)
        net_pnl = raw_pnl - comm
        pnl_pct = ((price - entry_price) / entry_price) * 100 if direction == "LONG" else ((entry_price - price) / entry_price) * 100
        
        self.capital += self.position["size"] + net_pnl

        self.trades.append({
            "entry":         entry_price,
            "exit":          price,
            "direction":     direction,
            "size":          round(self.position["size"], 2),
            "qty":           round(qty, 6),
            "pnl":           round(net_pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "rr":            round(pnl_pct / 2.0, 2),  # relative reference stop
            "held":          self.position["held"],
            "reason_entry":  self.position["setup"],
            "reason_exit":   reason,
            "regime":        self.position["regime"],
            "conf":          self.position["conf"],
            "entry_time":    self.position["time"],
            "exit_time":     ts,
            "entry_signals": self.position["entry_signals"],
            "exit_signals":  [s["name"] for s in signals],
        })

        self.position = None
        self.state    = "COOLDOWN"
        self.cooldown_rem = COOLDOWN
        self._update_equity(price)
        return reason

    def _update_equity(self, price):
        eq = self.capital
        if self.position:
            entry = self.position["entry"]
            qty = self.position["qty"]
            if self.position["direction"] == "LONG":
                eq += (price - entry) * qty
            else:
                eq += (entry - price) * qty
        self.equity_curve.append(round(eq, 2))
        if eq > self.peak_capital:
            self.peak_capital = eq
        dd = ((self.peak_capital - eq) / self.peak_capital) * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def force_close(self, price, ts):
        if self.position:
            self._close(price, ts, "END", [], {})

    def stats(self):
        t = self.trades
        if not t: return {}
        wins    = [x for x in t if x["pnl"] > 0]
        losses  = [x for x in t if x["pnl"] <= 0]
        pnls    = [x["pnl"] for x in t]
        gross_w = sum(x["pnl"] for x in wins)  if wins   else 0
        gross_l = abs(sum(x["pnl"] for x in losses)) if losses else 0
        profit_factor = gross_w / gross_l if gross_l > 0 else float('inf')

        returns = np.array(pnls) / INITIAL_CAP
        sharpe  = (returns.mean() / returns.std() * math.sqrt(len(returns))
                   if returns.std() > 0 else 0)

        return {
            "total_trades": len(t),
            "win_rate":     round(len(wins) / len(t) * 100, 1),
            "net_profit":   round(self.capital - INITIAL_CAP, 2),
            "profit_factor":round(profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "consec_wins":  len(wins),
            "consec_losses":len(losses)
        }


def run():
    print(f"\n{'='*72}")
    print("  ANTIGRAVITY OPTIMIZED DEEP BACKTEST — Performance Verification")
    print(f"{'='*72}\n")

    candles = fetch_candles()
    print(f"[*] Fetched {len(candles)} candles | "
          f"{candles[0]['timestamp']} → {candles[-1]['timestamp']}")

    ml_tracker = OptimizedTracker("ML Engine")
    vd_tracker = OptimizedTracker("Verdict Engine")

    processed = 0
    for i in range(LOOKBACK, len(candles) - 1):
        if processed >= TEST_CANDLES:
            break
        window       = candles[:i+1]
        next_candle  = candles[i+1]
        cur_price    = window[-1]["close"]
        ts           = window[-1]["timestamp"]

        indicators   = compute_all_indicators(window)
        signals      = detect_signals(window, indicators, INTERVAL)
        verdict      = verdict_calc(signals)

        buy_conf  = signals[0].get("buy_confluence",  0) if signals else 0
        sell_conf = signals[0].get("sell_confluence", 0) if signals else 0

        # Extract indicators
        vol_ratio_list = indicators.get("volume_ratio", [])
        vol_ratio = vol_ratio_list[-1]["value"] if isinstance(vol_ratio_list, list) and vol_ratio_list else 1.0

        ml_signal_dir = next(
            (s["direction"] for s in signals if s["name"] == "AI ML Prediction"), None)
        
        # ML Engine setups
        ml_enter_long = False
        ml_enter_short = False
        if vol_ratio >= 1.5:  # Enforce P1 volume filter
            if ml_signal_dir == "BUY":
                conf_sigs = [s for s in signals
                             if s["direction"] == "BUY"
                             and s["name"] != "AI ML Prediction"
                             and s.get("signal_role") == "ENTRY"]
                if len(conf_sigs) >= 1:
                    ml_enter_long = True
            elif ml_signal_dir == "SELL":
                conf_sigs = [s for s in signals
                             if s["direction"] == "SELL"
                             and s["name"] != "AI ML Prediction"
                             and s.get("signal_role") == "ENTRY"]
                if len(conf_sigs) >= 1:
                    ml_enter_short = True

        ml_says_buy = (ml_signal_dir == "BUY")
        ml_says_sell = (ml_signal_dir == "SELL")
        ml_exit = False
        if ml_tracker.position:
            if ml_tracker.position["direction"] == "LONG":
                exit_patterns = any(s["name"] in LONG_EXIT_PATTERNS for s in signals if s["direction"] == "SELL")
                ml_exit = ml_says_sell or exit_patterns
            else:
                exit_patterns = any(s["name"] in SHORT_EXIT_PATTERNS for s in signals if s["direction"] == "BUY")
                ml_exit = ml_says_buy or exit_patterns

        ml_tracker.on_candle(ml_enter_long, ml_enter_short, ml_exit, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

        # Verdict Engine setups
        vd_enter_long = False
        vd_enter_short = False
        if vol_ratio >= 1.5:  # Enforce P1 volume filter
            if verdict in ("BUY", "STRONG BUY") and buy_conf >= MIN_CONF:
                vd_enter_long = True
            elif verdict in ("SELL", "STRONG SELL") and sell_conf >= MIN_CONF:
                vd_enter_short = True

        vd_exit = False
        if vd_tracker.position:
            if vd_tracker.position["direction"] == "LONG":
                vd_exit = verdict in ("SELL", "STRONG SELL")
            else:
                vd_exit = verdict in ("BUY", "STRONG BUY")

        vd_tracker.on_candle(vd_enter_long, vd_enter_short, vd_exit, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

        processed += 1

    final_price = candles[-1]["close"]
    final_ts    = candles[-1]["timestamp"]
    ml_tracker.force_close(final_price, final_ts)
    vd_tracker.force_close(final_price, final_ts)

    bnh = ((final_price - candles[LOOKBACK]["close"]) / candles[LOOKBACK]["close"]) * 100

    print(f"[✓] Optimized deep backtest complete.")
    print(f"\n    ML Engine (Optimized):      ${ml_tracker.capital:,.2f}  "
          f"({(ml_tracker.capital-INITIAL_CAP)/INITIAL_CAP*100:+.2f}%)  "
          f"Trades: {len(ml_tracker.trades)}  "
          f"Profit Factor: {ml_tracker.stats().get('profit_factor',0)}")
    print(f"    Verdict Engine (Optimized): ${vd_tracker.capital:,.2f}  "
          f"({(vd_tracker.capital-INITIAL_CAP)/INITIAL_CAP*100:+.2f}%)  "
          f"Trades: {len(vd_tracker.trades)}  "
          f"Profit Factor: {vd_tracker.stats().get('profit_factor',0)}")
    print(f"    Buy & Hold:                 {bnh:.2f}%\n")


if __name__ == "__main__":
    run()
