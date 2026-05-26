"""
BTCUSDT 1H Backtest v2 — State-machine trading with buy→sell pairing.
Walk-forward simulation: at each candle, generate signals, then check if next candle confirms.
Generates a full P/L report.

Architecture matches live_monitor.py:
  IDLE ──(strong entry)──► IN_POSITION ──(exit signal/SL/TP)──► COOLDOWN ──(timer)──► IDLE
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
from datetime import datetime
from backend.indicators.calculator import compute_all_indicators
from backend.signals.detector import detect_signals
from backend.signals.probability_engine import compute_probability

# ─── CONFIG ───
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 500  # ~20 days of 1h candles
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 10.0  # 10% of capital per trade
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 4.0
COMMISSION_PCT = 0.1
LOOKBACK = 60  # minimum candles needed before generating signals

# ─── STATE MACHINE CONSTANTS ───
MIN_HOLD_CANDLES = 3      # Minimum 3 candles before signal-based exit
COOLDOWN_CANDLES = 2      # Wait 2 candles after exit before re-entry
MIN_ENTRY_CONFLUENCE = 6  # Minimum confluence score to enter a position

# Reversal patterns that trigger exit from a LONG position
LONG_EXIT_PATTERNS = {
    "Bearish Engulfing", "Evening Star", "Dark Cloud Cover",
    "Head and Shoulders", "Double Top", "MACD Bearish Cross",
    "EMA 9/21 Death Cross", "Three White Soldiers (Exhaustion)",
    "Descending Triangle", "Bollinger Rejection", "RSI Overbought"
}


def fetch_candles():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(url, params=params)
    data = resp.json()
    candles = []
    for k in data:
        candles.append({
            "timestamp": datetime.fromtimestamp(k[0]/1000).strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]), "quote_volume": float(k[7]),
            "trades": int(k[8])
        })
    return candles


def verdict_calc(signals):
    """Same logic as scanner.py _get_verdict — requires clear directional edge."""
    if not signals:
        return "NEUTRAL"
    
    actionable = [s for s in signals if s.get("direction") in ("BUY", "SELL")]
    if len(actionable) < 2:
        return "NEUTRAL"
    
    buy = sum(s["strength"] for s in actionable if s["direction"] == "BUY")
    sell = sum(s["strength"] for s in actionable if s["direction"] == "SELL")
    
    gap = abs(buy - sell)
    if gap < 3:
        return "NEUTRAL"
    
    if buy > sell + 6: return "STRONG BUY"
    if buy > sell + 3: return "BUY"
    if sell > buy + 6: return "STRONG SELL"
    if sell > buy + 3: return "SELL"
    return "NEUTRAL"


class BacktestTracker:
    """State-machine position tracker for backtesting."""
    
    def __init__(self, name, capital):
        self.name = name
        self.capital = capital
        self.position = None  # {entry, qty, size, time, candles_held}
        self.trades = []
        self.state = "IDLE"  # IDLE | IN_POSITION | COOLDOWN
        self.cooldown_remaining = 0
    
    def on_candle(self, should_enter, sell_signal_names, current_price, timestamp, 
                  buy_confluence, sell_confluence):
        """
        Process one candle. Returns trade event string or None.
        
        Args:
            should_enter: True if entry conditions are met
            sell_signal_names: list of SELL signal names for exit logic
            current_price: current close price
            timestamp: candle timestamp
            buy_confluence: total buy strength
            sell_confluence: total sell strength
        """
        if self.state == "COOLDOWN":
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = "IDLE"
            return None
        
        if self.state == "IN_POSITION":
            self.position["candles_held"] += 1
            
            # 1. Always check SL/TP (no minimum hold for risk management)
            pnl_pct = ((current_price - self.position["entry"]) / self.position["entry"]) * 100
            if pnl_pct <= -STOP_LOSS_PCT:
                return self._close(current_price, timestamp, "STOP_LOSS")
            elif pnl_pct >= TAKE_PROFIT_PCT:
                return self._close(current_price, timestamp, "TAKE_PROFIT")
            
            # 2. Signal-based exit only after minimum hold time
            if self.position["candles_held"] >= MIN_HOLD_CANDLES:
                # Exit if sell confluence strongly dominates
                if sell_confluence > buy_confluence + 4:
                    return self._close(current_price, timestamp, "REVERSAL")
                
                # Exit if reversal pattern detected
                for sig_name in sell_signal_names:
                    if sig_name in LONG_EXIT_PATTERNS:
                        return self._close(current_price, timestamp, "SIGNAL")
                
                # Exit if ML flips bearish
                if "AI ML Prediction" in sell_signal_names and sell_confluence > buy_confluence:
                    return self._close(current_price, timestamp, "ML_REVERSAL")
            
            return None
        
        if self.state == "IDLE":
            if should_enter:
                size = self.capital * (POSITION_SIZE_PCT / 100)
                qty = size / current_price
                self.capital -= size * (COMMISSION_PCT / 100)
                self.position = {
                    "entry": current_price, "qty": qty, "size": size,
                    "time": timestamp, "candles_held": 0
                }
                self.state = "IN_POSITION"
                return "ENTER"
        
        return None
    
    def _close(self, price, timestamp, reason):
        pnl = (price - self.position["entry"]) * self.position["qty"]
        commission = abs(pnl) * (COMMISSION_PCT / 100)
        net_pnl = pnl - commission
        self.capital += self.position["size"] + net_pnl
        pnl_pct = ((price - self.position["entry"]) / self.position["entry"]) * 100
        
        self.trades.append({
            "entry": self.position["entry"], "exit": price,
            "pnl": round(net_pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "reason": reason, "entry_time": self.position["time"],
            "exit_time": timestamp, "candles_held": self.position["candles_held"]
        })
        self.position = None
        self.state = "COOLDOWN"
        self.cooldown_remaining = COOLDOWN_CANDLES
        return reason
    
    def force_close(self, price, timestamp):
        """Close position at end of backtest."""
        if self.position:
            self._close(price, timestamp, "END")


def run_backtest():
    print(f"\n{'='*70}")
    print(f"  BTCUSDT 1H BACKTEST v2 — State Machine Walk-Forward")
    print(f"{'='*70}\n")
    
    candles = fetch_candles()
    print(f"[*] Fetched {len(candles)} candles from Binance")
    print(f"[*] Period: {candles[0]['timestamp']} to {candles[-1]['timestamp']}")
    print(f"[*] Price range: ${min(c['low'] for c in candles):,.2f} - ${max(c['high'] for c in candles):,.2f}")
    print(f"[*] State machine: MIN_HOLD={MIN_HOLD_CANDLES}, COOLDOWN={COOLDOWN_CANDLES}, MIN_CONFLUENCE={MIN_ENTRY_CONFLUENCE}\n")

    # ─── TRACKING ───
    ml_tracker = BacktestTracker("ML", INITIAL_CAPITAL)
    verdict_tracker = BacktestTracker("Verdict", INITIAL_CAPITAL)
    
    ml_predictions = []
    verdict_predictions = []
    pattern_detections = []
    
    for i in range(LOOKBACK, len(candles) - 1):
        window = candles[:i+1]
        next_candle = candles[i+1]
        current_price = window[-1]["close"]
        next_price = next_candle["close"]
        actual_move = "UP" if next_price > current_price else "DOWN"
        pct_move = ((next_price - current_price) / current_price) * 100
        
        indicators = compute_all_indicators(window)
        signals = detect_signals(window, indicators, INTERVAL)
        v = verdict_calc(signals)
        
        # Extract confluence scores
        buy_confluence = signals[0].get("buy_confluence", 0) if signals else 0
        sell_confluence = signals[0].get("sell_confluence", 0) if signals else 0
        
        # Extract ML prediction
        ml_pred = None
        for s in signals:
            if "ml_prob" in s:
                ml_pred = s["ml_prob"]
                break
        
        ml_signal_dir = None
        for s in signals:
            if s["name"] == "AI ML Prediction":
                ml_signal_dir = s["direction"]
                break
        
        if ml_pred is None:
            from backend.ml.predictor import predictor
            ml_result = predictor.predict(indicators, window, INTERVAL)
            if ml_result.get("status") == "ok":
                ml_pred = {"up": ml_result["up_prob"], "down": ml_result["down_prob"]}
        
        # Collect SELL signal names for exit logic
        sell_signal_names = [s["name"] for s in signals if s.get("direction") == "SELL"]
        
        # ─── ML PREDICTION TRACKING ───
        if ml_pred:
            ml_dir = "BUY" if ml_pred["up"] > 55 else ("SELL" if ml_pred["down"] > 55 else "NEUTRAL")
            correct = (ml_dir == "BUY" and actual_move == "UP") or (ml_dir == "SELL" and actual_move == "DOWN")
            ml_predictions.append({
                "time": window[-1]["timestamp"],
                "prediction": ml_dir,
                "up_prob": ml_pred["up"],
                "down_prob": ml_pred["down"],
                "actual": actual_move,
                "correct": correct,
                "pct_move": pct_move
            })
            
            # ML entry: needs ML BUY signal + at least 1 confirming technical signal
            ml_should_enter = False
            if ml_signal_dir == "BUY":
                confirming = [s for s in signals 
                             if s["direction"] == "BUY" 
                             and s["name"] != "AI ML Prediction"
                             and s.get("signal_role") == "ENTRY"]
                if len(confirming) >= 1:
                    ml_should_enter = True
            
            ml_tracker.on_candle(
                ml_should_enter, sell_signal_names, current_price,
                window[-1]["timestamp"], buy_confluence, sell_confluence
            )
        
        # ─── VERDICT TRACKING ───
        v_dir = "BUY" if v in ("BUY", "STRONG BUY") else ("SELL" if v in ("SELL", "STRONG SELL") else "NEUTRAL")
        correct_v = (v_dir == "BUY" and actual_move == "UP") or (v_dir == "SELL" and actual_move == "DOWN")
        verdict_predictions.append({
            "time": window[-1]["timestamp"],
            "verdict": v,
            "direction": v_dir,
            "actual": actual_move,
            "correct": correct_v,
            "pct_move": pct_move,
            "signal_count": len(signals)
        })
        
        # Verdict entry: needs BUY/STRONG BUY verdict + minimum confluence
        v_should_enter = (v_dir == "BUY" and buy_confluence >= MIN_ENTRY_CONFLUENCE)
        verdict_tracker.on_candle(
            v_should_enter, sell_signal_names, current_price,
            window[-1]["timestamp"], buy_confluence, sell_confluence
        )
        
        # Track pattern detections
        for s in signals:
            if s.get("type") in ("chart", "candlestick"):
                pattern_detections.append({
                    "time": window[-1]["timestamp"], "name": s["name"],
                    "type": s["type"], "direction": s["direction"],
                    "actual_next": actual_move,
                    "correct": (s["direction"] == "BUY" and actual_move == "UP") or 
                               (s["direction"] == "SELL" and actual_move == "DOWN")
                })
    
    # Close open positions
    final_price = candles[-1]["close"]
    ml_tracker.force_close(final_price, candles[-1]["timestamp"])
    verdict_tracker.force_close(final_price, candles[-1]["timestamp"])

    # ─── PRINT REPORT ───
    print(f"{'='*70}")
    print(f"  [REPORT] AI/ML PREDICTION ENGINE")
    print(f"{'='*70}")
    
    ml_actionable = [p for p in ml_predictions if p["prediction"] != "NEUTRAL"]
    ml_correct = [p for p in ml_actionable if p["correct"]]
    print(f"  Total predictions: {len(ml_predictions)}")
    print(f"  Actionable (BUY/SELL): {len(ml_actionable)}")
    print(f"  Correct: {len(ml_correct)}")
    print(f"  Accuracy: {len(ml_correct)/len(ml_actionable)*100:.1f}%" if ml_actionable else "  Accuracy: N/A")
    print(f"\n  [P/L] ML Trading Results (State Machine):")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${ml_tracker.capital:,.2f}")
    print(f"  Total P/L: ${ml_tracker.capital - INITIAL_CAPITAL:,.2f} ({(ml_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%)")
    print(f"  Total Trades: {len(ml_tracker.trades)}")
    if ml_tracker.trades:
        wins = [t for t in ml_tracker.trades if t["pnl"] > 0]
        losses = [t for t in ml_tracker.trades if t["pnl"] <= 0]
        print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"  Win Rate: {len(wins)/len(ml_tracker.trades)*100:.1f}%")
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in wins]):,.2f}" if wins else "  Avg Win: $0")
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losses]):,.2f}" if losses else "  Avg Loss: $0")
        print(f"  Avg Hold: {np.mean([t['candles_held'] for t in ml_tracker.trades]):.1f} candles")
    
    print(f"\n{'='*70}")
    print(f"  [REPORT] VERDICT (CONFLUENCE) ENGINE")
    print(f"{'='*70}")
    
    v_actionable = [p for p in verdict_predictions if p["direction"] != "NEUTRAL"]
    v_correct = [p for p in v_actionable if p["correct"]]
    print(f"  Total predictions: {len(verdict_predictions)}")
    print(f"  Actionable (BUY/SELL): {len(v_actionable)}")
    print(f"  Correct: {len(v_correct)}")
    print(f"  Accuracy: {len(v_correct)/len(v_actionable)*100:.1f}%" if v_actionable else "  Accuracy: N/A")
    print(f"\n  [P/L] Verdict Trading Results (State Machine):")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${verdict_tracker.capital:,.2f}")
    print(f"  Total P/L: ${verdict_tracker.capital - INITIAL_CAPITAL:,.2f} ({(verdict_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%)")
    print(f"  Total Trades: {len(verdict_tracker.trades)}")
    if verdict_tracker.trades:
        wins = [t for t in verdict_tracker.trades if t["pnl"] > 0]
        losses = [t for t in verdict_tracker.trades if t["pnl"] <= 0]
        print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"  Win Rate: {len(wins)/len(verdict_tracker.trades)*100:.1f}%")
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in wins]):,.2f}" if wins else "  Avg Win: $0")
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losses]):,.2f}" if losses else "  Avg Loss: $0")
        print(f"  Avg Hold: {np.mean([t['candles_held'] for t in verdict_tracker.trades]):.1f} candles")
    
    print(f"\n{'='*70}")
    print(f"  [REPORT] PATTERN DETECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total patterns detected: {len(pattern_detections)}")
    
    # Group by pattern name
    pattern_stats = {}
    for p in pattern_detections:
        name = p["name"]
        if name not in pattern_stats:
            pattern_stats[name] = {"total": 0, "correct": 0, "type": p["type"]}
        pattern_stats[name]["total"] += 1
        if p["correct"]:
            pattern_stats[name]["correct"] += 1
    
    print(f"\n  {'Pattern':<30} {'Type':<12} {'Count':>6} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*30} {'-'*12} {'-'*6} {'-'*8} {'-'*9}")
    for name, stats in sorted(pattern_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        acc = stats["correct"]/stats["total"]*100 if stats["total"] else 0
        print(f"  {name:<30} {stats['type']:<12} {stats['total']:>6} {stats['correct']:>8} {acc:>8.1f}%")
    
    # ─── TRADE DETAILS ───
    print(f"\n{'='*70}")
    print(f"  [LOG] ML TRADE LOG")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'P/L':>10} {'%':>7} {'Hold':>5} {'Reason':<12}")
    print(f"  {'-'*3} {'-'*12} {'-'*12} {'-'*10} {'-'*7} {'-'*5} {'-'*12}")
    for idx, t in enumerate(ml_tracker.trades, 1):
        emoji = "W" if t["pnl"] > 0 else "L"
        print(f"  {idx:>3} ${t['entry']:>10,.2f} ${t['exit']:>10,.2f} ${t['pnl']:>8,.2f} {t['pnl_pct']:>6.2f}% {t['candles_held']:>4}h {t['reason']:<12} {emoji}")
    
    print(f"\n{'='*70}")
    print(f"  [LOG] VERDICT TRADE LOG")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'P/L':>10} {'%':>7} {'Hold':>5} {'Reason':<12}")
    print(f"  {'-'*3} {'-'*12} {'-'*12} {'-'*10} {'-'*7} {'-'*5} {'-'*12}")
    for idx, t in enumerate(verdict_tracker.trades, 1):
        emoji = "W" if t["pnl"] > 0 else "L"
        print(f"  {idx:>3} ${t['entry']:>10,.2f} ${t['exit']:>10,.2f} ${t['pnl']:>8,.2f} {t['pnl_pct']:>6.2f}% {t['candles_held']:>4}h {t['reason']:<12} {emoji}")
    
    # ─── BUY AND HOLD COMPARISON ───
    start_price = candles[LOOKBACK]["close"]
    end_price = candles[-1]["close"]
    bnh_return = ((end_price - start_price) / start_price) * 100
    print(f"\n{'='*70}")
    print(f"  [REPORT] BUY & HOLD COMPARISON")
    print(f"{'='*70}")
    print(f"  Buy & Hold Return: {bnh_return:.2f}%")
    print(f"  ML Strategy Return: {(ml_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%")
    print(f"  Verdict Strategy Return: {(verdict_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%")
    
    winner = "ML" if ml_tracker.capital > verdict_tracker.capital else "Verdict"
    print(f"\n  >>> Winner: {winner} Strategy")
    print(f"{'='*70}\n")
    
    # Save full results to JSON
    results = {
        "symbol": SYMBOL, "interval": INTERVAL, "period": f"{candles[0]['timestamp']} to {candles[-1]['timestamp']}",
        "config": {
            "min_hold_candles": MIN_HOLD_CANDLES,
            "cooldown_candles": COOLDOWN_CANDLES,
            "min_entry_confluence": MIN_ENTRY_CONFLUENCE,
            "stop_loss_pct": STOP_LOSS_PCT,
            "take_profit_pct": TAKE_PROFIT_PCT,
        },
        "ml_engine": {
            "accuracy": round(len(ml_correct)/len(ml_actionable)*100, 1) if ml_actionable else 0,
            "trades": ml_tracker.trades,
            "final_capital": round(ml_tracker.capital, 2),
            "pnl": round(ml_tracker.capital - INITIAL_CAPITAL, 2),
            "return_pct": round((ml_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2)
        },
        "verdict_engine": {
            "accuracy": round(len(v_correct)/len(v_actionable)*100, 1) if v_actionable else 0,
            "trades": verdict_tracker.trades,
            "final_capital": round(verdict_tracker.capital, 2),
            "pnl": round(verdict_tracker.capital - INITIAL_CAPITAL, 2),
            "return_pct": round((verdict_tracker.capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2)
        },
        "buy_and_hold_return_pct": round(bnh_return, 2),
        "pattern_stats": {k: {**v, "accuracy": round(v["correct"]/v["total"]*100, 1) if v["total"] else 0} for k, v in pattern_stats.items()}
    }
    
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Full report saved to: {report_path}")

if __name__ == "__main__":
    run_backtest()
