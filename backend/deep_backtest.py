"""
Deep Backtest v1 — Institutional-grade analysis for ML vs Verdict engines.
Collects every metric needed for the full comparative report.
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
LIMIT         = 1000          # ~41 days of 1h candles — rich dataset
INITIAL_CAP   = 10_000.0
POS_SIZE_PCT  = 10.0          # 10% of capital per trade
SL_PCT        = 2.0
TP_PCT        = 4.0
COMMISSION    = 0.1           # 0.1% per side
LOOKBACK      = 0             # start from first candle for short test
TEST_CANDLES = 2             # simulate exactly 2 candles (≈2 h)

MIN_HOLD      = 3
COOLDOWN      = 2
MIN_CONF      = 6

LONG_EXIT_PATTERNS = {
    "Bearish Engulfing", "Evening Star", "Dark Cloud Cover",
    "Head and Shoulders", "Double Top", "MACD Bearish Cross",
    "EMA 9/21 Death Cross", "Three White Soldiers (Exhaustion)",
    "Descending Triangle", "Bollinger Rejection", "RSI Overbought"
}


# ───────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────

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
    """0-100 score based on signal alignment and strength."""
    aligned   = [s for s in signals if s.get("direction") == direction]
    opposing  = [s for s in signals if s.get("direction") != direction and s.get("direction") in ("BUY","SELL")]
    a_strength = sum(s["strength"] for s in aligned)
    o_strength = sum(s["strength"] for s in opposing)
    total_str  = a_strength + o_strength if (a_strength + o_strength) > 0 else 1
    raw = (a_strength / total_str) * 100
    # bonus for multiple confirming signals
    if len(aligned) >= 3: raw = min(100, raw + 10)
    return round(raw, 1)


def market_regime(indicators):
    """Return trending / ranging / volatile based on latest ADX value."""
    # ADX indicator is stored as a list of {"time":..., "value":...}
    adx_data = indicators.get("adx", [])
    if isinstance(adx_data, list) and adx_data:
        adx = adx_data[-1].get("value", 0)
    else:
        adx = float(adx_data) if isinstance(adx_data, (int, float)) else 0
    if adx > 30:
        return "TRENDING"
    if adx > 18:
        return "MILD_TREND"
    return "RANGING"
    """Return trending / ranging / volatile."""
    adx = indicators.get("adx", 0)
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


# ───────────────────────────────────────────
#  STATE MACHINE TRACKER
# ───────────────────────────────────────────

class Tracker:
    def __init__(self, name):
        self.name              = name
        self.capital           = INITIAL_CAP
        self.position          = None
        self.trades            = []
        self.state             = "IDLE"
        self.cooldown_rem      = 0
        self.peak_capital      = INITIAL_CAP
        self.max_drawdown      = 0.0
        self.equity_curve      = [INITIAL_CAP]

    def on_candle(self, should_enter, sell_names, price, ts,
                  buy_conf, sell_conf, signals, indicators):
        if self.state == "COOLDOWN":
            self.cooldown_rem -= 1
            if self.cooldown_rem <= 0:
                self.state = "IDLE"
            self._update_equity(price)
            return None

        if self.state == "IN_POSITION":
            self.position["held"] += 1
            pnl_pct = ((price - self.position["entry"]) / self.position["entry"]) * 100

            if pnl_pct <= -SL_PCT:
                return self._close(price, ts, "STOP_LOSS", signals, indicators)
            if pnl_pct >= TP_PCT:
                return self._close(price, ts, "TAKE_PROFIT", signals, indicators)

            if self.position["held"] >= MIN_HOLD:
                if sell_conf > buy_conf + 4:
                    return self._close(price, ts, "REVERSAL", signals, indicators)
                for n in sell_names:
                    if n in LONG_EXIT_PATTERNS:
                        return self._close(price, ts, "SIGNAL", signals, indicators)
                if "AI ML Prediction" in sell_names and sell_conf > buy_conf:
                    return self._close(price, ts, "ML_REVERSAL", signals, indicators)

            self._update_equity(price)
            return None

        if self.state == "IDLE" and should_enter:
            size = self.capital * (POS_SIZE_PCT / 100)
            qty  = size / price
            self.capital -= size * (COMMISSION / 100)
            conf = confidence_score(signals, "BUY")
            regime = market_regime(indicators)
            setup  = classify_setup(signals, "BUY")
            self.position = {
                "entry": price, "qty": qty, "size": size,
                "time": ts, "held": 0,
                "conf": conf, "regime": regime, "setup": setup,
                "entry_signals": [s["name"] for s in signals]
            }
            self.state = "IN_POSITION"
            self._update_equity(price)
            return "ENTER"

        self._update_equity(price)
        return None

    def _close(self, price, ts, reason, signals, indicators):
        raw_pnl  = (price - self.position["entry"]) * self.position["qty"]
        comm     = abs(raw_pnl) * (COMMISSION / 100)
        net_pnl  = raw_pnl - comm
        pnl_pct  = ((price - self.position["entry"]) / self.position["entry"]) * 100
        rr       = abs(pnl_pct / SL_PCT) if pnl_pct > 0 else -abs(pnl_pct / SL_PCT)
        self.capital += self.position["size"] + net_pnl

        self.trades.append({
            "entry":         self.position["entry"],
            "exit":          price,
            "direction":     "LONG",
            "size":          round(self.position["size"], 2),
            "qty":           round(self.position["qty"], 6),
            "pnl":           round(net_pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "rr":            round(rr, 2),
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
            eq += (price - self.position["entry"]) * self.position["qty"]
        self.equity_curve.append(round(eq, 2))
        if eq > self.peak_capital:
            self.peak_capital = eq
        dd = ((self.peak_capital - eq) / self.peak_capital) * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def force_close(self, price, ts):
        if self.position:
            self._close(price, ts, "END", [], {})

    # ── analytics ──────────────────────────────

    def stats(self):
        t = self.trades
        if not t:
            return {}
        wins    = [x for x in t if x["pnl"] > 0]
        losses  = [x for x in t if x["pnl"] <= 0]
        pnls    = [x["pnl"] for x in t]
        gross_w = sum(x["pnl"] for x in wins)  if wins   else 0
        gross_l = abs(sum(x["pnl"] for x in losses)) if losses else 0

        returns = np.array(pnls) / INITIAL_CAP
        sharpe  = (returns.mean() / returns.std() * math.sqrt(len(returns))
                   if returns.std() > 0 else 0)

        consec_wins = consec_losses = cur_w = cur_l = max_w = max_l = 0
        for x in t:
            if x["pnl"] > 0:
                cur_w += 1; cur_l = 0
            else:
                cur_l += 1; cur_w = 0
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)

        regime_wins = {}
        for x in t:
            r = x["regime"]
            regime_wins.setdefault(r, {"w": 0, "l": 0})
            if x["pnl"] > 0: regime_wins[r]["w"] += 1
            else:             regime_wins[r]["l"] += 1

        setup_wins = {}
        for x in t:
            s = x["reason_entry"]
            setup_wins.setdefault(s, {"w": 0, "l": 0, "pnl": 0})
            if x["pnl"] > 0: setup_wins[s]["w"] += 1
            else:             setup_wins[s]["l"] += 1
            setup_wins[s]["pnl"] += x["pnl"]

        return {
            "total_trades":    len(t),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        round(len(wins)/len(t)*100, 1),
            "final_capital":   round(self.capital, 2),
            "net_pnl":         round(self.capital - INITIAL_CAP, 2),
            "return_pct":      round((self.capital - INITIAL_CAP)/INITIAL_CAP*100, 2),
            "max_drawdown":    round(self.max_drawdown, 2),
            "profit_factor":   round(gross_w/gross_l, 3) if gross_l else float("inf"),
            "avg_win":         round(np.mean([x["pnl"] for x in wins]), 2) if wins else 0,
            "avg_loss":        round(np.mean([x["pnl"] for x in losses]), 2) if losses else 0,
            "avg_hold":        round(np.mean([x["held"] for x in t]), 1),
            "avg_rr":          round(np.mean([x["rr"] for x in t]), 2),
            "sharpe_est":      round(sharpe, 3),
            "max_consec_wins": max_w,
            "max_consec_loss": max_l,
            "avg_conf":        round(np.mean([x["conf"] for x in t]), 1),
            "regime_breakdown":regime_wins,
            "setup_breakdown": setup_wins,
            "best_trade":      max(t, key=lambda x: x["pnl"]),
            "worst_trade":     min(t, key=lambda x: x["pnl"]),
        }


# ───────────────────────────────────────────
#  MAIN BACKTEST
# ───────────────────────────────────────────

def run():
    print(f"\n{'='*72}")
    print("  ANTIGRAVITY DEEP BACKTEST — Institutional Analysis Report")
    print(f"{'='*72}\n")

    candles = fetch_candles()
    print(f"[*] Fetched {len(candles)} candles | "
          f"{candles[0]['timestamp']} → {candles[-1]['timestamp']}")

    ml_tracker = Tracker("ML Engine")
    vd_tracker = Tracker("Verdict Engine")

    processed = 0
    for i in range(LOOKBACK, len(candles) - 1):
        if processed >= TEST_CANDLES:
            break
        window       = candles[:i+1]
        next_candle  = candles[i+1]
        cur_price    = window[-1]["close"]
        nxt_price    = next_candle["close"]
        ts           = window[-1]["timestamp"]

        indicators   = compute_all_indicators(window)
        signals      = detect_signals(window, indicators, INTERVAL)
        verdict      = verdict_calc(signals)

        buy_conf  = signals[0].get("buy_confluence",  0) if signals else 0
        sell_conf = signals[0].get("sell_confluence", 0) if signals else 0

        ml_signal_dir = next(
            (s["direction"] for s in signals if s["name"] == "AI ML Prediction"), None)
        sell_names = [s["name"] for s in signals if s.get("direction") == "SELL"]

        # ML entry: ML BUY + at least 1 confirming ENTRY signal
        ml_enter = False
        if ml_signal_dir == "BUY":
            conf_sigs = [s for s in signals
                         if s["direction"] == "BUY"
                         and s["name"] != "AI ML Prediction"
                         and s.get("signal_role") == "ENTRY"]
            if len(conf_sigs) >= 1:
                ml_enter = True

        ml_tracker.on_candle(ml_enter, sell_names, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

        # Verdict entry: BUY/STRONG BUY + minimum confluence
        v_dir     = "BUY" if verdict in ("BUY","STRONG BUY") else (
                    "SELL" if verdict in ("SELL","STRONG SELL") else "NEUTRAL")
        vd_enter  = (v_dir == "BUY" and buy_conf >= MIN_CONF)
        vd_tracker.on_candle(vd_enter, sell_names, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

        processed += 1

        window       = candles[:i+1]
        next_candle  = candles[i+1]
        cur_price    = window[-1]["close"]
        nxt_price    = next_candle["close"]
        ts           = window[-1]["timestamp"]

        indicators   = compute_all_indicators(window)
        signals      = detect_signals(window, indicators, INTERVAL)
        verdict      = verdict_calc(signals)

        buy_conf  = signals[0].get("buy_confluence",  0) if signals else 0
        sell_conf = signals[0].get("sell_confluence", 0) if signals else 0

        ml_signal_dir = next(
            (s["direction"] for s in signals if s["name"] == "AI ML Prediction"), None)
        sell_names = [s["name"] for s in signals if s.get("direction") == "SELL"]

        # ML entry: ML BUY + at least 1 confirming ENTRY signal
        ml_enter = False
        if ml_signal_dir == "BUY":
            conf_sigs = [s for s in signals
                         if s["direction"] == "BUY"
                         and s["name"] != "AI ML Prediction"
                         and s.get("signal_role") == "ENTRY"]
            if len(conf_sigs) >= 1:
                ml_enter = True

        ml_tracker.on_candle(ml_enter, sell_names, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

        # Verdict entry: BUY/STRONG BUY + minimum confluence
        v_dir     = "BUY" if verdict in ("BUY","STRONG BUY") else (
                    "SELL" if verdict in ("SELL","STRONG SELL") else "NEUTRAL")
        vd_enter  = (v_dir == "BUY" and buy_conf >= MIN_CONF)
        vd_tracker.on_candle(vd_enter, sell_names, cur_price, ts,
                              buy_conf, sell_conf, signals, indicators)

    final_price = candles[-1]["close"]
    final_ts    = candles[-1]["timestamp"]
    ml_tracker.force_close(final_price, final_ts)
    vd_tracker.force_close(final_price, final_ts)

    bnh = ((final_price - candles[LOOKBACK]["close"]) / candles[LOOKBACK]["close"]) * 100

    output = {
        "meta": {
            "symbol":   SYMBOL,
            "interval": INTERVAL,
            "candles":  len(candles),
            "period":   f"{candles[0]['timestamp']} → {candles[-1]['timestamp']}",
            "config": {
                "initial_capital":    INITIAL_CAP,
                "position_size_pct":  POS_SIZE_PCT,
                "stop_loss_pct":      SL_PCT,
                "take_profit_pct":    TP_PCT,
                "commission_pct":     COMMISSION,
                "min_hold_candles":   MIN_HOLD,
                "cooldown_candles":   COOLDOWN,
                "min_entry_conf":     MIN_CONF,
            }
        },
        "buy_and_hold_pct":   round(bnh, 2),
        "ml_engine": {
            "stats":        ml_tracker.stats(),
            "trades":       ml_tracker.trades,
            "equity_curve": ml_tracker.equity_curve,
        },
        "verdict_engine": {
            "stats":        vd_tracker.stats(),
            "trades":       vd_tracker.trades,
            "equity_curve": vd_tracker.equity_curve,
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "deep_backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Deep backtest complete. Results saved to: {out_path}")
    print(f"\n    ML Engine:      ${ml_tracker.capital:,.2f}  "
          f"({(ml_tracker.capital-INITIAL_CAP)/INITIAL_CAP*100:.1f}%)  "
          f"{ml_tracker.stats().get('total_trades',0)} trades")
    print(f"    Verdict Engine: ${vd_tracker.capital:,.2f}  "
          f"({(vd_tracker.capital-INITIAL_CAP)/INITIAL_CAP*100:.1f}%)  "
          f"{vd_tracker.stats().get('total_trades',0)} trades")
    print(f"    Buy & Hold:     {bnh:.2f}%\n")

if __name__ == "__main__":
    run()
