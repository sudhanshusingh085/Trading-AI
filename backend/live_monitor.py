"""
Live 2-Hour Multi-Timeframe Monitor — Tracks BTCUSDT live predictions from the running app.
Uses 1-minute base candles for setup/execution, and 5-minute candles for higher timeframe trend filtering.
Supports both LONG and SHORT trades, adapting dynamically to bullish/bearish market conditions.

Resolves 3 Core Strategy Weaknesses:
  1. ATR-Based Stops: Replaces fixed SL/TP with dynamic 1.5x ATR stops and 3.0x ATR targets (guaranteed 2:1 R:R).
  2. Increased Hold Time: Raises MIN_HOLD_POLLS to 30 (5 minutes) to protect trades from early signal noise.
  3. Volume & Profitability confirmation: Enforces volume_ratio >= 1.5 and minimum volatility threshold.
"""
import time
import requests
import os
from datetime import datetime

# ─── CONFIG ───
API_URL_1M = "http://127.0.0.1:8000/api/crypto/BTCUSDT/analysis?interval=1m&limit=100"
API_URL_5M = "http://127.0.0.1:8000/api/crypto/BTCUSDT/analysis?interval=5m&limit=100"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_test_report.md")
DURATION_SECONDS = 7200  # 2 hours
POLL_INTERVAL = 10       # 10 seconds
HTTP_TIMEOUT = 15        # robust 15-second timeout for Binance fetch latency

INITIAL_CAPITAL = 10000.0
POSITION_SIZE = 1000.0   # $1000 per trade

# ─── STATE MACHINE CONSTANTS ───
MIN_HOLD_POLLS = 30      # Minimum 30 polls (5 minutes) before signal-based exit to filters noise
COOLDOWN_POLLS = 6       # Cooldown 1 minute after exit to prevent whipsaws
MIN_ENTRY_CONFLUENCE = 6  # Minimum confluence score to enter a position

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


class StrategyTracker:
    """
    State-machine based strategy tracker implementing bidirectional (LONG/SHORT) trading
    with dynamic ATR stops, minimum hold constraint, and volume confirmation checks.
    """
    
    def __init__(self, name):
        self.name = name
        self.capital = INITIAL_CAPITAL
        self.position = None  # {entry_price, qty, timestamp, polls_held, direction, sl_price, tp_price}
        self.trades = []      # list of closed trade dicts
        self.state = "IDLE"   # IDLE | IN_POSITION | COOLDOWN
        self.cooldown_remaining = 0

    def execute_logic(self, entry_signal, exit_signal, current_price, timestamp, latest_atr, volume_ratio, higher_tf_ok_long=True, higher_tf_ok_short=True):
        """
        Executes strategy logic with ATR stops, volume confirmations, and long hold.
        """
        if self.state == "COOLDOWN":
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = "IDLE"
            return None
        
        if self.state == "IN_POSITION":
            self.position["polls_held"] += 1
            
            direction = self.position["direction"]
            sl_price = self.position["sl_price"]
            tp_price = self.position["tp_price"]
            
            # 1. ATR-based risk management (SL / TP boundary check)
            if direction == "LONG":
                if current_price <= sl_price:
                    return self.close_position(sl_price, timestamp, "STOP_LOSS")
                if current_price >= tp_price:
                    return self.close_position(tp_price, timestamp, "TAKE_PROFIT")
            else:  # SHORT
                if current_price >= sl_price:
                    return self.close_position(sl_price, timestamp, "STOP_LOSS")
                if current_price <= tp_price:
                    return self.close_position(tp_price, timestamp, "TAKE_PROFIT")
            
            # 2. Signal-based exit (enforced minimum hold time to filters noise)
            if self.position["polls_held"] >= MIN_HOLD_POLLS and exit_signal:
                return self.close_position(current_price, timestamp, "EXIT_SIGNAL")
            
            return None  # Keep holding
        
        if self.state == "IDLE":
            # P1 Volume Confirmation Filter
            if volume_ratio < 1.5:
                return None
                
            # Minimum expected volatility filter (ensure ATR is large enough to outrun commission)
            if latest_atr < current_price * 0.0005:
                return None

            if entry_signal == "BUY" and higher_tf_ok_long:
                sl_dist = 1.5 * latest_atr
                tp_dist = 3.0 * latest_atr
                self.position = {
                    "entry_price": current_price,
                    "qty": POSITION_SIZE / current_price,
                    "timestamp": timestamp,
                    "polls_held": 0,
                    "direction": "LONG",
                    "sl_price": current_price - sl_dist,
                    "tp_price": current_price + tp_dist
                }
                self.state = "IN_POSITION"
                return f"ENTER LONG @ ${current_price:,.2f} (SL: ${self.position['sl_price']:,.2f} | TP: ${self.position['tp_price']:,.2f})"
                
            elif entry_signal == "SELL" and higher_tf_ok_short:
                sl_dist = 1.5 * latest_atr
                tp_dist = 3.0 * latest_atr
                self.position = {
                    "entry_price": current_price,
                    "qty": POSITION_SIZE / current_price,
                    "timestamp": timestamp,
                    "polls_held": 0,
                    "direction": "SHORT",
                    "sl_price": current_price + sl_dist,
                    "tp_price": current_price - tp_dist
                }
                self.state = "IN_POSITION"
                return f"ENTER SHORT @ ${current_price:,.2f} (SL: ${self.position['sl_price']:,.2f} | TP: ${self.position['tp_price']:,.2f})"
        
        return None

    def close_position(self, exit_price, timestamp, reason):
        entry_price = self.position["entry_price"]
        qty = self.position["qty"]
        direction = self.position["direction"]
        
        if direction == "LONG":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl = (entry_price - exit_price) * qty
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        trade = {
            "direction": direction,
            "entry_time": self.position["timestamp"],
            "exit_time": timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "hold_polls": self.position["polls_held"]
        }
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
        self.state = "COOLDOWN"
        self.cooldown_remaining = COOLDOWN_POLLS
        return f"🔴 EXIT {direction} @ ${exit_price:,.2f} ({reason}) | P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"


def _extract_entry_signal_ml(active_signals, buy_confluence, sell_confluence):
    ml_direction = None
    for s in active_signals:
        if s["name"] == "AI ML Prediction":
            ml_direction = s["direction"]
            break
            
    if not ml_direction:
        return None
        
    confirming_signals = [s for s in active_signals 
                          if s["direction"] == ml_direction 
                          and s["name"] != "AI ML Prediction"
                          and s.get("signal_role") == "ENTRY"]
                          
    if len(confirming_signals) >= 1:
        return ml_direction
        
    return None


def _extract_entry_signal_verdict(verdict, buy_confluence, sell_confluence):
    if verdict in ("BUY", "STRONG BUY") and buy_confluence >= MIN_ENTRY_CONFLUENCE:
        return "BUY"
    if verdict in ("SELL", "STRONG SELL") and sell_confluence >= MIN_ENTRY_CONFLUENCE:
        return "SELL"
    return None


def generate_report(elapsed, remaining, current_price, last_update, active_signals, verdict, 
                    higher_tf_verdict, higher_tf_signals, ml_tracker, verdict_tracker, log_entries):
    ml_pnl = ml_tracker.capital - INITIAL_CAPITAL
    verdict_pnl = verdict_tracker.capital - INITIAL_CAPITAL
    
    ml_live_pnl = 0
    if ml_tracker.position:
        entry_price = ml_tracker.position["entry_price"]
        qty = ml_tracker.position["qty"]
        if ml_tracker.position["direction"] == "LONG":
            ml_live_pnl = (current_price - entry_price) * qty
        else:
            ml_live_pnl = (entry_price - current_price) * qty
        
    v_live_pnl = 0
    if verdict_tracker.position:
        entry_price = verdict_tracker.position["entry_price"]
        qty = verdict_tracker.position["qty"]
        if verdict_tracker.position["direction"] == "LONG":
            v_live_pnl = (current_price - entry_price) * qty
        else:
            v_live_pnl = (entry_price - current_price) * qty

    # Trend filter alignment status
    higher_tf_ok_long = higher_tf_verdict not in ("SELL", "STRONG SELL")
    higher_tf_ok_short = higher_tf_verdict not in ("BUY", "STRONG BUY")
    
    filter_status = "🔄 BIDIRECTIONAL (Longs and Shorts permitted)"
    if not higher_tf_ok_long and higher_tf_ok_short:
        filter_status = "🐻 BEARISH ALIGNED (Only Short entries permitted)"
    elif higher_tf_ok_long and not higher_tf_ok_short:
        filter_status = "🐂 BULLISH ALIGNED (Only Long entries permitted)"
    elif not higher_tf_ok_long and not higher_tf_ok_short:
        filter_status = "🔴 BLOCKED (Highly volatile/restricted)"

    report = f"""# 🔴 LIVE 2-HOUR MULTI-TIMEFRAME TRADING AI TEST REPORT

This report updates in real-time every 10 seconds. Open this file in VS Code (`Ctrl+Shift+V` or `Cmd+Shift+V` for Preview) to watch the live simulation!

- **Status:** Running 🟢
- **Time Elapsed:** {elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s / 2h 00m 00s
- **Last Updated:** {last_update}
- **Current BTCUSDT Price:** **${current_price:,.2f}**

---

## 🌐 Multi-Timeframe Trend Alignment (Base: 1m | Filter: 5m)
- **Base Timeframe (1m) Confluence Verdict:** `{verdict}`
- **Filter Timeframe (5m) Trend Verdict:** `{higher_tf_verdict}`
- **Trend Filter Status:** **{filter_status}**
- **Active 5m Signals:** {", ".join([f"`{s['name']}` ({s['direction']})" for s in higher_tf_signals]) if higher_tf_signals else "None (Neutral)"}

---

## 📊 Live Base Signal Status (1m setups)
- **Active 1m Signals Detected:**
{chr(10).join([f"  - `{s['name']}` ({s['direction']} | strength: {s['strength']} | role: {s.get('signal_role', 'ENTRY')})" for s in active_signals]) if active_signals else "  - None (Neutral)"}

---

## 🤖 Strategy 1: AI/ML Engine [{ml_tracker.state}]
- **Account Balance:** ${ml_tracker.capital:,.2f}
- **Net Realized P/L:** ${ml_pnl:+.2f} ({ml_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not ml_tracker.position else f"{ml_tracker.position['direction']} from ${ml_tracker.position['entry_price']:,.2f} (Hold: {ml_tracker.position['polls_held']} polls | SL: ${ml_tracker.position['sl_price']:,.2f} | TP: ${ml_tracker.position['tp_price']:,.2f} | Open P/L: ${ml_live_pnl:+.2f})"}
- **Closed Trades:** {len(ml_tracker.trades)}

### ML Trade Log
| Direction | Entry Time | Exit Time | Entry Price | Exit Price | P/L ($) | P/L (%) | Reason |
|---|---|---|---|---|---|---|---|
"""
    if not ml_tracker.trades:
        report += "| - | - | - | - | - | - | - | - |\n"
    for t in ml_tracker.trades:
        report += f"| {t['direction']} | {t['entry_time']} | {t['exit_time']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {t['pnl']:+.2f} | {t['pnl_pct']:+.2f}% | {t['reason']} |\n"

    report += f"""
---

## ⚖️ Strategy 2: Confluence Verdict Engine [{verdict_tracker.state}]
- **Account Balance:** ${verdict_tracker.capital:,.2f}
- **Net Realized P/L:** ${verdict_pnl:+.2f} ({verdict_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not verdict_tracker.position else f"{verdict_tracker.position['direction']} from ${verdict_tracker.position['entry_price']:,.2f} (Hold: {verdict_tracker.position['polls_held']} polls | SL: ${verdict_tracker.position['sl_price']:,.2f} | TP: ${verdict_tracker.position['tp_price']:,.2f} | Open P/L: ${v_live_pnl:+.2f})"}
- **Closed Trades:** {len(verdict_tracker.trades)}

### Verdict Trade Log
| Direction | Entry Time | Exit Time | Entry Price | Exit Price | P/L ($) | P/L (%) | Reason |
|---|---|---|---|---|---|---|---|
"""
    if not verdict_tracker.trades:
        report += "| - | - | - | - | - | - | - | - |\n"
    for t in verdict_tracker.trades:
        report += f"| {t['direction']} | {t['entry_time']} | {t['exit_time']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {t['pnl']:+.2f} | {t['pnl_pct']:+.2f}% | {t['reason']} |\n"

    report += """
---

## 📋 Activity Log (Last 15 events)
"""
    for entry in log_entries[-15:]:
        report += f"- {entry}\n"
        
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    print("[*] Starting live 2-hour BTCUSDT optimized bidirectional multi-timeframe test monitor...")
    ml_tracker = StrategyTracker("AI/ML Engine")
    verdict_tracker = StrategyTracker("Verdict Engine")
    
    start_time = time.time()
    log_entries = ["System initialized. Dynamic ATR stops, 5m hold constraints, and volume filters active."]
    
    # Pre-populate report
    generate_report(0, DURATION_SECONDS, 0.0, "Initializing...", [], "INITIALIZING", "INITIALIZING", [], ml_tracker, verdict_tracker, log_entries)

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed >= DURATION_SECONDS:
            break
            
        remaining = DURATION_SECONDS - elapsed
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Fetch with robust 15-second timeout
            resp_1m = requests.get(API_URL_1M, timeout=HTTP_TIMEOUT)
            resp_5m = requests.get(API_URL_5M, timeout=HTTP_TIMEOUT)
            
            if resp_1m.status_code == 200 and resp_5m.status_code == 200:
                data_1m = resp_1m.json()
                data_5m = resp_5m.json()
                
                current_price = data_1m["latest_price"]
                verdict_1m = data_1m["verdict"]
                active_signals_1m = data_1m["signals"]
                timestamp = data_1m["candles"][-1]["timestamp"]
                
                # Extract ATR and Volume Ratio indicators from 1m context
                indicators_1m = data_1m.get("indicators", {})
                atr_list = indicators_1m.get("atr", [])
                vol_ratio_list = indicators_1m.get("volume_ratio", [])
                
                latest_atr = atr_list[-1]["value"] if atr_list else current_price * 0.002
                volume_ratio = vol_ratio_list[-1]["value"] if vol_ratio_list else 1.0
                
                verdict_5m = data_5m["verdict"]
                active_signals_5m = data_5m["signals"]
                
                # Higher timeframe trend filter rules
                higher_tf_ok_long = verdict_5m not in ("SELL", "STRONG SELL")
                higher_tf_ok_short = verdict_5m not in ("BUY", "STRONG BUY")
                
                # Extract base timeframe (1m) confluence metrics
                buy_confluence = active_signals_1m[0].get("buy_confluence", 0) if active_signals_1m else 0
                sell_confluence = active_signals_1m[0].get("sell_confluence", 0) if active_signals_1m else 0
                
                # Exit signal extraction
                ml_says_buy = any(s["name"] == "AI ML Prediction" and s["direction"] == "BUY" for s in active_signals_1m)
                ml_says_sell = any(s["name"] == "AI ML Prediction" and s["direction"] == "SELL" for s in active_signals_1m)
                
                # ─── ML ENGINE SETUP ───
                ml_entry = _extract_entry_signal_ml(active_signals_1m, buy_confluence, sell_confluence)
                ml_exit = False
                if ml_tracker.position:
                    if ml_tracker.position["direction"] == "LONG":
                        exit_patterns = any(s["name"] in LONG_EXIT_PATTERNS for s in active_signals_1m if s["direction"] == "SELL")
                        ml_exit = ml_says_sell or exit_patterns
                    else:
                        exit_patterns = any(s["name"] in SHORT_EXIT_PATTERNS for s in active_signals_1m if s["direction"] == "BUY")
                        ml_exit = ml_says_buy or exit_patterns
                        
                ml_event = ml_tracker.execute_logic(
                    ml_entry, ml_exit, current_price, timestamp, latest_atr, volume_ratio,
                    higher_tf_ok_long=higher_tf_ok_long, higher_tf_ok_short=higher_tf_ok_short
                )
                if ml_event:
                    log_entries.append(f"[{last_update}] [ML] {ml_event}")
                    
                # ─── VERDICT ENGINE SETUP ───
                v_entry = _extract_entry_signal_verdict(verdict_1m, buy_confluence, sell_confluence)
                v_exit = False
                if verdict_tracker.position:
                    if verdict_tracker.position["direction"] == "LONG":
                        v_exit = verdict_1m in ("SELL", "STRONG SELL")
                    else:
                        v_exit = verdict_1m in ("BUY", "STRONG BUY")
                        
                v_event = verdict_tracker.execute_logic(
                    v_entry, v_exit, current_price, timestamp, latest_atr, volume_ratio,
                    higher_tf_ok_long=higher_tf_ok_long, higher_tf_ok_short=higher_tf_ok_short
                )
                if v_event:
                    log_entries.append(f"[{last_update}] [VERDICT] {v_event}")
                    
                # Update report
                generate_report(elapsed, remaining, current_price, last_update, active_signals_1m, verdict_1m, 
                                verdict_5m, active_signals_5m, ml_tracker, verdict_tracker, log_entries)
                
            else:
                log_entries.append(f"[{last_update}] Backend error: 1m HTTP {resp_1m.status_code} | 5m HTTP {resp_5m.status_code}")
        except Exception as e:
            log_entries.append(f"[{last_update}] Fetch failed: {str(e)}")
            
        time.sleep(POLL_INTERVAL)

    # End of test
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entries.append(f"[{last_update}] Live 2-hour multi-timeframe test finished successfully.")
    
    # Close any remaining open positions at current final price
    try:
        resp = requests.get(API_URL_1M, timeout=HTTP_TIMEOUT)
        final_price = resp.json()["latest_price"] if resp.status_code == 200 else ml_tracker.position["entry_price"]
    except:
        final_price = 0.0
        
    if ml_tracker.position and final_price > 0:
        ml_tracker.close_position(final_price, last_update, "TEST_END")
    if verdict_tracker.position and final_price > 0:
        verdict_tracker.close_position(final_price, last_update, "TEST_END")
        
    # Final report update
    generate_report(DURATION_SECONDS, 0, final_price, last_update, [], "FINISHED", "FINISHED", [], ml_tracker, verdict_tracker, log_entries)
    print("[*] Live test complete. Report updated.")


if __name__ == "__main__":
    main()
