"""
Live 1-Hour Monitor — Tracks BTCUSDT live predictions from the running app.
Uses 1-minute candles so we get actionable data within the 1-hour window.
Polls every 10 seconds, executes virtual trades, tracks P/L, and updates a live Markdown report.
"""
import time
import requests
import os
from datetime import datetime

# ─── CONFIG ───
API_URL = "http://127.0.0.1:8000/api/crypto/BTCUSDT/analysis?interval=1m&limit=100"
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_test_report.md")
DURATION_SECONDS = 3600  # 1 hour
POLL_INTERVAL = 10       # 10 seconds

INITIAL_CAPITAL = 10000.0
POSITION_SIZE = 1000.0   # $1000 per trade
STOP_LOSS_PCT = 1.0      # 1% stop loss (tightened for 1m timeframe)
TAKE_PROFIT_PCT = 2.0    # 2% take profit

class StrategyTracker:
    def __init__(self, name):
        self.name = name
        self.capital = INITIAL_CAPITAL
        self.position = None # {entry_price, qty, timestamp}
        self.trades = [] # list of closed trade dicts

    def execute_logic(self, signal, current_price, timestamp):
        trade_event = None
        
        # Check SL/TP if in position
        if self.position:
            pnl_pct = ((current_price - self.position["entry_price"]) / self.position["entry_price"]) * 100
            if pnl_pct <= -STOP_LOSS_PCT:
                trade_event = self.close_position(current_price, timestamp, "STOP_LOSS")
            elif pnl_pct >= TAKE_PROFIT_PCT:
                trade_event = self.close_position(current_price, timestamp, "TAKE_PROFIT")

        # Process standard entry/exit signals if SL/TP didn't fire
        if not trade_event:
            if signal == "BUY" and not self.position:
                self.position = {
                    "entry_price": current_price,
                    "qty": POSITION_SIZE / current_price,
                    "timestamp": timestamp
                }
                trade_event = f"🟢 ENTER LONG @ ${current_price:,.2f}"
            elif signal == "SELL" and self.position:
                trade_event = self.close_position(current_price, timestamp, "SIGNAL")
                
        return trade_event

    def close_position(self, current_price, timestamp, reason):
        entry_price = self.position["entry_price"]
        qty = self.position["qty"]
        pnl = (current_price - entry_price) * qty
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        trade = {
            "entry_time": self.position["timestamp"],
            "exit_time": timestamp,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason
        }
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
        return f"🔴 EXIT LONG @ ${current_price:,.2f} ({reason}) | P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"

def generate_report(elapsed, remaining, current_price, last_update, active_signals, verdict, ml_tracker, verdict_tracker, log_entries):
    ml_pnl = ml_tracker.capital - INITIAL_CAPITAL
    verdict_pnl = verdict_tracker.capital - INITIAL_CAPITAL
    
    # Calculate live unreleased P/L
    ml_live_pnl = 0
    if ml_tracker.position:
        ml_live_pnl = (current_price - ml_tracker.position["entry_price"]) * ml_tracker.position["qty"]
        
    v_live_pnl = 0
    if verdict_tracker.position:
        v_live_pnl = (current_price - verdict_tracker.position["entry_price"]) * verdict_tracker.position["qty"]

    report = f"""# 🔴 LIVE 1-HOUR TRADING AI TEST REPORT

This report is updating in real-time every 10 seconds. Open this file in VS Code (`Ctrl+Shift+V` or `Cmd+Shift+V` for Preview) to watch the live simulation!

- **Status:** Running 🟢
- **Time Elapsed:** {elapsed // 60}m {elapsed % 60}s / 60m 00s
- **Last Updated:** {last_update}
- **Current BTCUSDT Price:** **${current_price:,.2f}**

---

## 📊 Live Signal Status
- **Current Confluence Verdict:** `{verdict}`
- **Active Signals Detected:**
{chr(10).join([f"  - `{s['name']}` ({s['direction']} | strength: {s['strength']})" for s in active_signals]) if active_signals else "  - None (Neutral)"}

---

## 🤖 Strategy 1: AI/ML Engine
- **Account Balance:** ${ml_tracker.capital:,.2f}
- **Net Realized P/L:** ${ml_pnl:+.2f} ({ml_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not ml_tracker.position else f"LONG from ${ml_tracker.position['entry_price']:,.2f} (Current Open P/L: ${ml_live_pnl:+.2f})"}
- **Closed Trades:** {len(ml_tracker.trades)}

### ML Trade Log
| Entry Time | Exit Time | Entry Price | Exit Price | P/L ($) | P/L (%) | Reason |
|---|---|---|---|---|---|---|
"""
    if not ml_tracker.trades:
        report += "| - | - | - | - | - | - | - |\n"
    for t in ml_tracker.trades:
        report += f"| {t['entry_time']} | {t['exit_time']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {t['pnl']:+.2f} | {t['pnl_pct']:+.2f}% | {t['reason']} |\n"

    report += f"""
---

## ⚖️ Strategy 2: Confluence Verdict Engine
- **Account Balance:** ${verdict_tracker.capital:,.2f}
- **Net Realized P/L:** ${verdict_pnl:+.2f} ({verdict_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not verdict_tracker.position else f"LONG from ${verdict_tracker.position['entry_price']:,.2f} (Current Open P/L: ${v_live_pnl:+.2f})"}
- **Closed Trades:** {len(verdict_tracker.trades)}

### Verdict Trade Log
| Entry Time | Exit Time | Entry Price | Exit Price | P/L ($) | P/L (%) | Reason |
|---|---|---|---|---|---|---|
"""
    if not verdict_tracker.trades:
        report += "| - | - | - | - | - | - | - |\n"
    for t in verdict_tracker.trades:
        report += f"| {t['entry_time']} | {t['exit_time']} | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {t['pnl']:+.2f} | {t['pnl_pct']:+.2f}% | {t['reason']} |\n"

    report += """
---

## 📋 Activity Log (Last 15 events)
"""
    for entry in log_entries[-15:]:
        report += f"- {entry}\n"
        
    with open(REPORT_PATH, "w") as f:
        f.write(report)

def main():
    print("[*] Starting live 1-hour BTCUSDT test monitor...")
    ml_tracker = StrategyTracker("AI/ML Engine")
    verdict_tracker = StrategyTracker("Verdict Engine")
    
    start_time = time.time()
    log_entries = ["System initialized. Monitoring live Binance feed via backend API..."]
    
    # Pre-populate report
    generate_report(0, DURATION_SECONDS, 0.0, "Initializing...", [], "INITIALIZING", ml_tracker, verdict_tracker, log_entries)

    while True:
        elapsed = int(time.time() - start_time)
        if elapsed >= DURATION_SECONDS:
            break
            
        remaining = DURATION_SECONDS - elapsed
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            resp = requests.get(API_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                current_price = data["latest_price"]
                verdict = data["verdict"]
                active_signals = data["signals"]
                timestamp = data["candles"][-1]["timestamp"]
                
                # Extract ML signal direction
                ml_signal_dir = None
                for s in active_signals:
                    if s["name"] == "AI ML Prediction":
                        ml_signal_dir = s["direction"]
                        break
                
                # Verdict signal direction
                v_signal_dir = None
                if verdict in ("BUY", "STRONG BUY"):
                    v_signal_dir = "BUY"
                elif verdict in ("SELL", "STRONG SELL"):
                    v_signal_dir = "SELL"
                
                # Execute Strategy Logic
                ml_event = ml_tracker.execute_logic(ml_signal_dir, current_price, timestamp)
                if ml_event:
                    log_entries.append(f"[{last_update}] [ML] {ml_event}")
                    
                v_event = verdict_tracker.execute_logic(v_signal_dir, current_price, timestamp)
                if v_event:
                    log_entries.append(f"[{last_update}] [VERDICT] {v_event}")
                    
                # Update report
                generate_report(elapsed, remaining, current_price, last_update, active_signals, verdict, ml_tracker, verdict_tracker, log_entries)
                
            else:
                log_entries.append(f"[{last_update}] Backend error: HTTP {resp.status_code}")
        except Exception as e:
            log_entries.append(f"[{last_update}] Fetch failed: {str(e)}")
            
        time.sleep(POLL_INTERVAL)

    # End of test
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entries.append(f"[{last_update}] Live 1-hour test finished successfully.")
    
    # Close any remaining open positions at current final price
    try:
        resp = requests.get(API_URL, timeout=5)
        final_price = resp.json()["latest_price"] if resp.status_code == 200 else ml_tracker.position["entry_price"]
    except:
        final_price = 0.0
        
    if ml_tracker.position and final_price > 0:
        ml_tracker.close_position(final_price, last_update, "TEST_END")
    if verdict_tracker.position and final_price > 0:
        verdict_tracker.close_position(final_price, last_update, "TEST_END")
        
    # Final report update
    generate_report(DURATION_SECONDS, 0, final_price, last_update, [], "FINISHED", ml_tracker, verdict_tracker, log_entries)
    print("[*] Live test complete. Report updated.")

if __name__ == "__main__":
    main()
