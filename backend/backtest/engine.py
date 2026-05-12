"""
Backtesting Engine — Test strategies against historical data.
Tracks entries, exits, P&L, win rate, drawdown, Sharpe ratio.
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional
from backend.indicators.calculator import compute_all_indicators


def run_backtest(
    candles: list[dict],
    strategy_fn: Callable,
    initial_capital: float = 100000,
    position_size_pct: float = 10.0,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 5.0,
    commission_pct: float = 0.1,
) -> dict:
    """
    Run a backtest on historical candle data.

    Args:
        candles: OHLCV candle data
        strategy_fn: Function(candles, indicators, i) -> "BUY" | "SELL" | None
        initial_capital: Starting capital
        position_size_pct: % of capital per trade
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        commission_pct: Commission per trade

    Returns:
        Dict with trades, metrics, equity curve
    """
    indicators = compute_all_indicators(candles)
    capital = initial_capital
    position = None
    trades = []
    equity_curve = []

    for i in range(50, len(candles)):
        price = candles[i]["close"]
        ts = candles[i]["timestamp"]

        # Check stop loss / take profit if in position
        if position:
            pnl_pct = ((price - position["entry_price"]) / position["entry_price"]) * 100
            if position["direction"] == "SHORT":
                pnl_pct = -pnl_pct

            if pnl_pct <= -stop_loss_pct:
                capital, trade = _close_position(position, price, ts, capital, commission_pct, "STOP_LOSS")
                trades.append(trade)
                position = None
            elif pnl_pct >= take_profit_pct:
                capital, trade = _close_position(position, price, ts, capital, commission_pct, "TAKE_PROFIT")
                trades.append(trade)
                position = None

        # Get strategy signal
        signal = strategy_fn(candles[:i+1], indicators, i)

        if signal == "BUY" and position is None:
            size = capital * (position_size_pct / 100)
            qty = size / price
            commission = size * (commission_pct / 100)
            capital -= commission
            position = {
                "direction": "LONG",
                "entry_price": price,
                "entry_time": ts,
                "quantity": qty,
                "size": size,
            }
        elif signal == "SELL" and position and position["direction"] == "LONG":
            capital, trade = _close_position(position, price, ts, capital, commission_pct, "SIGNAL")
            trades.append(trade)
            position = None

        equity = capital + (position["quantity"] * price if position else 0)
        equity_curve.append({"time": ts, "value": round(equity, 2)})

    # Close any open position at the end
    if position:
        price = candles[-1]["close"]
        capital, trade = _close_position(position, price, candles[-1]["timestamp"], capital, commission_pct, "END")
        trades.append(trade)

    return {
        "trades": trades,
        "metrics": _compute_metrics(trades, initial_capital, capital, equity_curve),
        "equity_curve": equity_curve,
    }


def _close_position(position, price, ts, capital, commission_pct, reason):
    pnl = (price - position["entry_price"]) * position["quantity"]
    commission = abs(pnl) * (commission_pct / 100)
    net_pnl = pnl - commission
    capital += position["size"] + net_pnl

    trade = {
        "direction": position["direction"],
        "entry_price": round(position["entry_price"], 2),
        "exit_price": round(price, 2),
        "entry_time": position["entry_time"],
        "exit_time": ts,
        "quantity": round(position["quantity"], 6),
        "pnl": round(net_pnl, 2),
        "pnl_pct": round((net_pnl / position["size"]) * 100, 2),
        "exit_reason": reason,
    }
    return capital, trade


def _compute_metrics(trades, initial_capital, final_capital, equity_curve):
    if not trades:
        return {"error": "No trades executed"}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for point in equity_curve:
        if point["value"] > peak:
            peak = point["value"]
        dd = (peak - point["value"]) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe ratio (simplified)
    if len(trades) > 1:
        returns = [t["pnl_pct"] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "risk_reward_ratio": round(risk_reward, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "initial_capital": initial_capital,
        "final_capital": round(final_capital, 2),
        "return_pct": round(((final_capital - initial_capital) / initial_capital) * 100, 2),
    }
