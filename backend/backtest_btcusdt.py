"""
BTCUSDT 1H Backtest — Tests both AI/ML predictions and Verdict signals.
Walk-forward simulation: at each candle, generate signals, then check if next candle confirms.
Generates a full P/L report.
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

def verdict(signals):
    if not signals: return "NEUTRAL"
    buy = sum(s["strength"] for s in signals if s["direction"] == "BUY")
    sell = sum(s["strength"] for s in signals if s["direction"] == "SELL")
    if buy > sell + 3: return "STRONG BUY"
    if buy > sell: return "BUY"
    if sell > buy + 3: return "STRONG SELL"
    if sell > buy: return "SELL"
    return "NEUTRAL"

def run_backtest():
    print(f"\n{'='*70}")
    print(f"  BTCUSDT 1H BACKTEST — Walk-Forward Simulation")
    print(f"{'='*70}\n")
    
    candles = fetch_candles()
    print(f"[*] Fetched {len(candles)} candles from Binance")
    print(f"[*] Period: {candles[0]['timestamp']} → {candles[-1]['timestamp']}")
    print(f"[*] Price range: ${min(c['low'] for c in candles):,.2f} — ${max(c['high'] for c in candles):,.2f}\n")

    # ─── TRACKING ───
    ml_trades = []        # trades based on ML prediction
    verdict_trades = []   # trades based on verdict
    ml_predictions = []   # all ML predictions for accuracy tracking
    verdict_predictions = []
    
    ml_capital = INITIAL_CAPITAL
    verdict_capital = INITIAL_CAPITAL
    ml_position = None
    verdict_position = None
    
    pattern_detections = []  # track pattern detections over time
    
    for i in range(LOOKBACK, len(candles) - 1):
        window = candles[:i+1]
        next_candle = candles[i+1]
        current_price = window[-1]["close"]
        next_price = next_candle["close"]
        actual_move = "UP" if next_price > current_price else "DOWN"
        pct_move = ((next_price - current_price) / current_price) * 100
        
        indicators = compute_all_indicators(window)
        signals = detect_signals(window, indicators, INTERVAL)
        v = verdict(signals)
        
        # Extract ML prediction
        ml_pred = None
        for s in signals:
            if "ml_prob" in s:
                ml_pred = s["ml_prob"]
                break
        
        # Also check for AI ML Prediction signal
        ml_signal_dir = None
        for s in signals:
            if s["name"] == "AI ML Prediction":
                ml_signal_dir = s["direction"]
                break
        
        # If no ml_prob attached, use the heuristic fallback info
        if ml_pred is None:
            # The ML predictor always returns something
            from backend.ml.predictor import predictor
            ml_result = predictor.predict(indicators, window, INTERVAL)
            if ml_result.get("status") == "ok":
                ml_pred = {"up": ml_result["up_prob"], "down": ml_result["down_prob"]}
        
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
            
            # ML Trading Logic
            if ml_position is None and ml_dir == "BUY":
                size = ml_capital * (POSITION_SIZE_PCT / 100)
                qty = size / current_price
                ml_capital -= size * (COMMISSION_PCT / 100)
                ml_position = {"entry": current_price, "qty": qty, "size": size, "time": window[-1]["timestamp"]}
            elif ml_position:
                pnl_pct = ((current_price - ml_position["entry"]) / ml_position["entry"]) * 100
                if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT or ml_dir == "SELL":
                    pnl = (current_price - ml_position["entry"]) * ml_position["qty"]
                    commission = abs(pnl) * (COMMISSION_PCT / 100)
                    net_pnl = pnl - commission
                    ml_capital += ml_position["size"] + net_pnl
                    reason = "STOP_LOSS" if pnl_pct <= -STOP_LOSS_PCT else ("TAKE_PROFIT" if pnl_pct >= TAKE_PROFIT_PCT else "SIGNAL")
                    ml_trades.append({"entry": ml_position["entry"], "exit": current_price, "pnl": round(net_pnl, 2), "pnl_pct": round(pnl_pct, 2), "reason": reason, "entry_time": ml_position["time"], "exit_time": window[-1]["timestamp"]})
                    ml_position = None
        
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
        
        # Verdict Trading Logic
        if verdict_position is None and v_dir == "BUY":
            size = verdict_capital * (POSITION_SIZE_PCT / 100)
            qty = size / current_price
            verdict_capital -= size * (COMMISSION_PCT / 100)
            verdict_position = {"entry": current_price, "qty": qty, "size": size, "time": window[-1]["timestamp"]}
        elif verdict_position:
            pnl_pct = ((current_price - verdict_position["entry"]) / verdict_position["entry"]) * 100
            if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT or v_dir == "SELL":
                pnl = (current_price - verdict_position["entry"]) * verdict_position["qty"]
                commission = abs(pnl) * (COMMISSION_PCT / 100)
                net_pnl = pnl - commission
                verdict_capital += verdict_position["size"] + net_pnl
                reason = "STOP_LOSS" if pnl_pct <= -STOP_LOSS_PCT else ("TAKE_PROFIT" if pnl_pct >= TAKE_PROFIT_PCT else "SIGNAL")
                verdict_trades.append({"entry": verdict_position["entry"], "exit": current_price, "pnl": round(net_pnl, 2), "pnl_pct": round(pnl_pct, 2), "reason": reason, "entry_time": verdict_position["time"], "exit_time": window[-1]["timestamp"]})
                verdict_position = None
        
        # Track pattern detections
        for s in signals:
            if s.get("type") in ("chart", "candlestick"):
                pattern_detections.append({"time": window[-1]["timestamp"], "name": s["name"], "type": s["type"], "direction": s["direction"], "actual_next": actual_move, "correct": (s["direction"] == "BUY" and actual_move == "UP") or (s["direction"] == "SELL" and actual_move == "DOWN")})
    
    # Close open positions
    final_price = candles[-1]["close"]
    if ml_position:
        pnl = (final_price - ml_position["entry"]) * ml_position["qty"]
        ml_capital += ml_position["size"] + pnl
        ml_trades.append({"entry": ml_position["entry"], "exit": final_price, "pnl": round(pnl, 2), "pnl_pct": round(((final_price - ml_position["entry"]) / ml_position["entry"]) * 100, 2), "reason": "END", "entry_time": ml_position["time"], "exit_time": candles[-1]["timestamp"]})
    if verdict_position:
        pnl = (final_price - verdict_position["entry"]) * verdict_position["qty"]
        verdict_capital += verdict_position["size"] + pnl
        verdict_trades.append({"entry": verdict_position["entry"], "exit": final_price, "pnl": round(pnl, 2), "pnl_pct": round(((final_price - verdict_position["entry"]) / verdict_position["entry"]) * 100, 2), "reason": "END", "entry_time": verdict_position["time"], "exit_time": candles[-1]["timestamp"]})

    # ─── PRINT REPORT ───
    print(f"{'='*70}")
    print(f"  📊 AI/ML PREDICTION ENGINE REPORT")
    print(f"{'='*70}")
    
    ml_actionable = [p for p in ml_predictions if p["prediction"] != "NEUTRAL"]
    ml_correct = [p for p in ml_actionable if p["correct"]]
    print(f"  Total predictions: {len(ml_predictions)}")
    print(f"  Actionable (BUY/SELL): {len(ml_actionable)}")
    print(f"  Correct: {len(ml_correct)}")
    print(f"  Accuracy: {len(ml_correct)/len(ml_actionable)*100:.1f}%" if ml_actionable else "  Accuracy: N/A")
    print(f"\n  💰 ML Trading Results:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${ml_capital:,.2f}")
    print(f"  Total P/L: ${ml_capital - INITIAL_CAPITAL:,.2f} ({(ml_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%)")
    print(f"  Total Trades: {len(ml_trades)}")
    if ml_trades:
        wins = [t for t in ml_trades if t["pnl"] > 0]
        losses = [t for t in ml_trades if t["pnl"] <= 0]
        print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"  Win Rate: {len(wins)/len(ml_trades)*100:.1f}%")
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in wins]):,.2f}" if wins else "  Avg Win: $0")
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losses]):,.2f}" if losses else "  Avg Loss: $0")
    
    print(f"\n{'='*70}")
    print(f"  📊 VERDICT (CONFLUENCE) ENGINE REPORT")
    print(f"{'='*70}")
    
    v_actionable = [p for p in verdict_predictions if p["direction"] != "NEUTRAL"]
    v_correct = [p for p in v_actionable if p["correct"]]
    print(f"  Total predictions: {len(verdict_predictions)}")
    print(f"  Actionable (BUY/SELL): {len(v_actionable)}")
    print(f"  Correct: {len(v_correct)}")
    print(f"  Accuracy: {len(v_correct)/len(v_actionable)*100:.1f}%" if v_actionable else "  Accuracy: N/A")
    print(f"\n  💰 Verdict Trading Results:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${verdict_capital:,.2f}")
    print(f"  Total P/L: ${verdict_capital - INITIAL_CAPITAL:,.2f} ({(verdict_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%)")
    print(f"  Total Trades: {len(verdict_trades)}")
    if verdict_trades:
        wins = [t for t in verdict_trades if t["pnl"] > 0]
        losses = [t for t in verdict_trades if t["pnl"] <= 0]
        print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"  Win Rate: {len(wins)/len(verdict_trades)*100:.1f}%")
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in wins]):,.2f}" if wins else "  Avg Win: $0")
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losses]):,.2f}" if losses else "  Avg Loss: $0")
    
    print(f"\n{'='*70}")
    print(f"  📊 PATTERN DETECTION ANALYSIS")
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
    print(f"  {'─'*30} {'─'*12} {'─'*6} {'─'*8} {'─'*9}")
    for name, stats in sorted(pattern_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        acc = stats["correct"]/stats["total"]*100 if stats["total"] else 0
        print(f"  {name:<30} {stats['type']:<12} {stats['total']:>6} {stats['correct']:>8} {acc:>8.1f}%")
    
    # ─── TRADE DETAILS ───
    print(f"\n{'='*70}")
    print(f"  📋 ML TRADE LOG")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'P/L':>10} {'%':>7} {'Reason':<12}")
    print(f"  {'─'*3} {'─'*12} {'─'*12} {'─'*10} {'─'*7} {'─'*12}")
    for idx, t in enumerate(ml_trades, 1):
        emoji = "✅" if t["pnl"] > 0 else "❌"
        print(f"  {idx:>3} ${t['entry']:>10,.2f} ${t['exit']:>10,.2f} ${t['pnl']:>8,.2f} {t['pnl_pct']:>6.2f}% {t['reason']:<12} {emoji}")
    
    print(f"\n{'='*70}")
    print(f"  📋 VERDICT TRADE LOG")
    print(f"{'='*70}")
    print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'P/L':>10} {'%':>7} {'Reason':<12}")
    print(f"  {'─'*3} {'─'*12} {'─'*12} {'─'*10} {'─'*7} {'─'*12}")
    for idx, t in enumerate(verdict_trades, 1):
        emoji = "✅" if t["pnl"] > 0 else "❌"
        print(f"  {idx:>3} ${t['entry']:>10,.2f} ${t['exit']:>10,.2f} ${t['pnl']:>8,.2f} {t['pnl_pct']:>6.2f}% {t['reason']:<12} {emoji}")
    
    # ─── BUY AND HOLD COMPARISON ───
    start_price = candles[LOOKBACK]["close"]
    end_price = candles[-1]["close"]
    bnh_return = ((end_price - start_price) / start_price) * 100
    print(f"\n{'='*70}")
    print(f"  📊 BUY & HOLD COMPARISON")
    print(f"{'='*70}")
    print(f"  Buy & Hold Return: {bnh_return:.2f}%")
    print(f"  ML Strategy Return: {(ml_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%")
    print(f"  Verdict Strategy Return: {(verdict_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.2f}%")
    
    winner = "ML" if ml_capital > verdict_capital else "Verdict"
    print(f"\n  🏆 Winner: {winner} Strategy")
    print(f"{'='*70}\n")
    
    # Save full results to JSON
    results = {
        "symbol": SYMBOL, "interval": INTERVAL, "period": f"{candles[0]['timestamp']} to {candles[-1]['timestamp']}",
        "ml_engine": {"accuracy": round(len(ml_correct)/len(ml_actionable)*100, 1) if ml_actionable else 0, "trades": ml_trades, "final_capital": round(ml_capital, 2), "pnl": round(ml_capital - INITIAL_CAPITAL, 2), "return_pct": round((ml_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2)},
        "verdict_engine": {"accuracy": round(len(v_correct)/len(v_actionable)*100, 1) if v_actionable else 0, "trades": verdict_trades, "final_capital": round(verdict_capital, 2), "pnl": round(verdict_capital - INITIAL_CAPITAL, 2), "return_pct": round((verdict_capital - INITIAL_CAPITAL)/INITIAL_CAPITAL*100, 2)},
        "buy_and_hold_return_pct": round(bnh_return, 2),
        "pattern_stats": {k: {**v, "accuracy": round(v["correct"]/v["total"]*100, 1) if v["total"] else 0} for k, v in pattern_stats.items()}
    }
    
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Full report saved to: {report_path}")

if __name__ == "__main__":
    run_backtest()
