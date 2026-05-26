"""
Live 1-Hour Monitor — Tracks BTCUSDT live predictions from the running app.
Uses 1-minute candles so we get actionable data within the 1-hour window.
Polls every 10 seconds, executes virtual trades, tracks P/L, and updates a live Markdown report.

Architecture: State Machine trading with buy→sell pairing
  IDLE ──(strong entry signal)──► IN_POSITION ──(exit signal/SL/TP)──► COOLDOWN ──(timer)──► IDLE
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
STOP_LOSS_PCT = 1.0      # 1% stop loss
TAKE_PROFIT_PCT = 2.0    # 2% take profit

# ─── STATE MACHINE CONSTANTS ───
MIN_HOLD_POLLS = 6       # Minimum 6 polls (60 seconds) before signal-based exit
COOLDOWN_POLLS = 4       # Wait 4 polls (40 seconds) after exit before re-entry
MIN_ENTRY_CONFLUENCE = 6  # Minimum confluence score to enter a position


class StrategyTracker:
    """
    State-machine based strategy tracker implementing buy→sell pairing.
    
    States:
      IDLE         - No position, looking for ENTRY signals
      IN_POSITION  - Holding a position, looking for EXIT signals / SL / TP
      COOLDOWN     - Just exited, waiting before re-entry to prevent whipsaw
    """
    
    def __init__(self, name):
        self.name = name
        self.capital = INITIAL_CAPITAL
        self.position = None  # {entry_price, qty, timestamp, polls_held}
        self.trades = []      # list of closed trade dicts
        self.state = "IDLE"   # IDLE | IN_POSITION | COOLDOWN
        self.cooldown_remaining = 0

    def execute_logic(self, entry_signal, sell_verdict, ml_says_sell, current_price, timestamp):
        """
        Simple state machine: BUY when signaled, HOLD, exit only when SELL is signaled.
        
        Args:
            entry_signal: "BUY" or None — should we enter?
            sell_verdict: True if verdict/ML explicitly says SELL now
            ml_says_sell: True if ML prediction flipped to SELL
            current_price: current market price
            timestamp: current timestamp
        """
        if self.state == "COOLDOWN":
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0:
                self.state = "IDLE"
            return None
        
        if self.state == "IN_POSITION":
            self.position["polls_held"] += 1
            
            # 1. Risk management: SL/TP always active
            pnl_pct = ((current_price - self.position["entry_price"]) / self.position["entry_price"]) * 100
            if pnl_pct <= -STOP_LOSS_PCT:
                return self.close_position(current_price, timestamp, "STOP_LOSS")
            if pnl_pct >= TAKE_PROFIT_PCT:
                return self.close_position(current_price, timestamp, "TAKE_PROFIT")
            
            # 2. Hold until explicit SELL (after minimum hold time)
            if self.position["polls_held"] >= MIN_HOLD_POLLS:
                if sell_verdict or ml_says_sell:
                    reason = "SELL_SIGNAL"
                    return self.close_position(current_price, timestamp, reason)
            
            return None  # Keep holding
        
        if self.state == "IDLE":
            if entry_signal == "BUY":
                self.position = {
                    "entry_price": current_price,
                    "qty": POSITION_SIZE / current_price,
                    "timestamp": timestamp,
                    "polls_held": 0
                }
                self.state = "IN_POSITION"
                return f"ENTER LONG @ ${current_price:,.2f}"
        
        return None

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
            "reason": reason,
            "hold_polls": self.position["polls_held"]
        }
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
        self.state = "COOLDOWN"
        self.cooldown_remaining = COOLDOWN_POLLS
        return f"🔴 EXIT LONG @ ${current_price:,.2f} ({reason}) | P/L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"


def _extract_entry_signal_ml(active_signals, buy_confluence, sell_confluence):
    """
    Determine if ML engine should enter based on ML prediction + confirming signals.
    Returns "BUY" or None.
    """
    ml_signal = None
    for s in active_signals:
        if s["name"] == "AI ML Prediction" and s["direction"] == "BUY":
            ml_signal = s
            break
    
    if not ml_signal:
        return None
    
    # ML needs at least 1 confirming technical signal
    confirming_signals = [s for s in active_signals 
                         if s["direction"] == "BUY" 
                         and s["name"] != "AI ML Prediction"
                         and s.get("signal_role") == "ENTRY"]
    
    if len(confirming_signals) >= 1:
        return "BUY"
    
    return None


def _extract_entry_signal_verdict(verdict, buy_confluence, sell_confluence):
    """
    Determine if verdict engine should enter based on verdict + minimum confluence.
    Returns "BUY" or None.
    """
    if verdict in ("BUY", "STRONG BUY") and buy_confluence >= MIN_ENTRY_CONFLUENCE:
        return "BUY"
    return None


def _extract_exit_signals(active_signals, direction="SELL"):
    """Extract names of signals that suggest exiting a long position."""
    exit_names = []
    for s in active_signals:
        if s.get("direction") == direction:
            exit_names.append(s["name"])
        elif s.get("signal_role") == "EXIT" and s.get("direction") == direction:
            exit_names.append(s["name"])
    return exit_names


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
{chr(10).join([f"  - `{s['name']}` ({s['direction']} | strength: {s['strength']} | role: {s.get('signal_role', 'ENTRY')})" for s in active_signals]) if active_signals else "  - None (Neutral)"}

---

## 🤖 Strategy 1: AI/ML Engine [{ml_tracker.state}]
- **Account Balance:** ${ml_tracker.capital:,.2f}
- **Net Realized P/L:** ${ml_pnl:+.2f} ({ml_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not ml_tracker.position else f"LONG from ${ml_tracker.position['entry_price']:,.2f} (Hold: {ml_tracker.position['polls_held']} polls | Open P/L: ${ml_live_pnl:+.2f})"}
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

## ⚖️ Strategy 2: Confluence Verdict Engine [{verdict_tracker.state}]
- **Account Balance:** ${verdict_tracker.capital:,.2f}
- **Net Realized P/L:** ${verdict_pnl:+.2f} ({verdict_pnl/INITIAL_CAPITAL*100:+.2f}%)
- **Active Position:** {"None" if not verdict_tracker.position else f"LONG from ${verdict_tracker.position['entry_price']:,.2f} (Hold: {verdict_tracker.position['polls_held']} polls | Open P/L: ${v_live_pnl:+.2f})"}
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
        
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

def main():
    print("[*] Starting live 1-hour BTCUSDT test monitor (v2 - state machine)...")
    ml_tracker = StrategyTracker("AI/ML Engine")
    verdict_tracker = StrategyTracker("Verdict Engine")
    
    start_time = time.time()
    log_entries = ["System initialized (v2). State machine trading with buy→sell pairing active."]
    
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
                
                # Extract confluence scores
                buy_confluence = active_signals[0].get("buy_confluence", 0) if active_signals else 0
                sell_confluence = active_signals[0].get("sell_confluence", 0) if active_signals else 0
                
                # Check if ML says SELL
                ml_says_sell = any(s["name"] == "AI ML Prediction" and s["direction"] == "SELL" for s in active_signals)
                
                # Check if verdict says SELL
                verdict_says_sell = verdict in ("SELL", "STRONG SELL")
                
                # ─── ML ENGINE: enter on ML BUY + confirming signal, exit on SELL ───
                ml_entry = _extract_entry_signal_ml(active_signals, buy_confluence, sell_confluence)
                ml_event = ml_tracker.execute_logic(
                    ml_entry, ml_says_sell, ml_says_sell, current_price, timestamp
                )
                if ml_event:
                    log_entries.append(f"[{last_update}] [ML] {ml_event}")
                    
                # ─── VERDICT ENGINE: enter on BUY verdict, exit on SELL verdict ───
                v_entry = _extract_entry_signal_verdict(verdict, buy_confluence, sell_confluence)
                v_event = verdict_tracker.execute_logic(
                    v_entry, verdict_says_sell, ml_says_sell, current_price, timestamp
                )
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
