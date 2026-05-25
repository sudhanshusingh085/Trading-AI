import pandas as pd
import numpy as np
import os

# Import pure pandas indicator helpers from the existing backend calculator
from backend.indicators.calculator import _ema, _rsi, _true_range, _adx
from backend.data.crypto_fetcher import get_historical_data_sync

def fetch_and_build_dataset(symbol="BTCUSDT", interval="1h", lookforward=15, target_pct=1.0):
    """
    Fetches historical data from Binance, computes scale-invariant relative features,
    adds trend trajectories (velocity, acceleration, lags), and uses dynamic Triple Barrier labeling.
    """
    print(f"[*] Fetching {symbol} data at {interval} interval from Binance...")
    data = get_historical_data_sync(symbol, interval=interval, limit=1000)
    
    if not data or "candles" not in data:
        print(f"[-] Failed to fetch data: {data.get('error') if data else 'Unknown error'}")
        return None
        
    df = pd.DataFrame(data["candles"])
    print(f"[*] Raw data shape: {df.shape}")

    # Ensure correct capitalization of column names for the ML pipeline
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)

    # Compute Features (Pure Pandas)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. Standard Oscillators
    df['RSI_14'] = _rsi(close, 14)
    
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd_line = ema_12 - ema_26
    macd_signal = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal
    df['MACDh_ratio'] = macd_hist / close * 100
    
    # 2. Relative EMA Differences (Scale-Invariant)
    df['EMA9_ratio'] = (close - _ema(close, 9)) / _ema(close, 9) * 100
    df['EMA21_ratio'] = (close - _ema(close, 21)) / _ema(close, 21) * 100
    df['EMA50_ratio'] = (close - _ema(close, 50)) / _ema(close, 50) * 100
    
    # 3. Bollinger Bands Features
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bbu = sma_20 + 2 * std_20
    bbl = sma_20 - 2 * std_20
    df['BBB_20_2.0'] = (bbu - bbl) / sma_20 * 100
    df['BBP_20_2.0'] = (close - bbl) / (bbu - bbl)
    
    # 4. Relative Volatility Features
    tr = _true_range(high, low, close)
    df['ATRr_ratio'] = tr.rolling(14).mean() / close * 100
    df['ADX_14'] = _adx(high, low, close, 14)
    
    # 5. Vol and Returns Features
    df['returns'] = (close - df['Open']) / df['Open'] * 100
    df['spread_pct'] = (high - low) / close * 100
    
    vol_sma = volume.rolling(20).mean()
    df['vol_ratio'] = volume / vol_sma

    # ─── ADD MOMENTUM TRAJECTORY VECTORS (Velocity, Acceleration, Lags) ───
    trajectory_cols = ['RSI_14', 'MACDh_ratio', 'BBP_20_2.0', 'returns', 'vol_ratio']
    for col in trajectory_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_velocity'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_acceleration'] = df[f'{col}_velocity'] - (df[f'{col}_lag1'] - df[f'{col}_lag2'])
    
    # Drop rows with NaN from indicator warm-up and lag shift
    df.dropna(inplace=True)

    # ─── DYNAMIC TRIPLE BARRIER LABELING (Volatility-Adjusted Stops) ───
    atr = df['ATRr_ratio']
    labels = []
    
    for i in range(len(df)):
        if i >= len(df) - lookforward:
            labels.append(np.nan)
            continue
            
        price_start = df['Close'].iloc[i]
        vol_pct = atr.iloc[i] / 100.0
        
        # dynamic stop and take-profit targets based on volatility
        upper_barrier = price_start * (1.0 + 1.5 * vol_pct)
        lower_barrier = price_start * (1.0 - 1.0 * vol_pct)
        
        target_val = 0
        for j in range(1, lookforward + 1):
            curr_high = df['High'].iloc[i + j]
            curr_low = df['Low'].iloc[i + j]
            
            # 1. Stop loss barrier check
            if curr_low <= lower_barrier:
                target_val = 0
                break
            # 2. Take profit barrier check
            if curr_high >= upper_barrier:
                target_val = 1
                break
                
        labels.append(target_val)
        
    df['target'] = labels
    df.dropna(subset=['target'], inplace=True)
    
    # Build complete feature list
    features = [
        'RSI_14', 'MACDh_ratio',
        'EMA9_ratio', 'EMA21_ratio', 'EMA50_ratio', 
        'BBB_20_2.0', 'BBP_20_2.0',
        'ATRr_ratio', 'ADX_14',
        'returns', 'spread_pct', 'vol_ratio'
    ]
    
    # Dynamically append all generated trajectory features
    for col in trajectory_cols:
        features += [f'{col}_lag1', f'{col}_lag2', f'{col}_velocity', f'{col}_acceleration']
        
    available_features = [f for f in features if f in df.columns]
    return df[available_features + ['target']].copy()

def build_universal_dataset(interval="1h", target_pct=1.0):
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']
    all_data = []
    
    lookforward = 15
    adjusted_target_pct = target_pct
    if interval == "15m":
        lookforward = 20
        adjusted_target_pct = 0.5
    elif interval == "1m":
        lookforward = 30
        adjusted_target_pct = 0.2
    elif interval == "1d":
        lookforward = 8
        adjusted_target_pct = 3.0
        
    print(f"[*] Building Universal Scale-Invariant Dataset with Triple-Barrier Labels for {interval}...")
    for sym in symbols:
        df = fetch_and_build_dataset(sym, interval, lookforward, adjusted_target_pct)
        if df is not None and not df.empty:
            all_data.append(df)
            
    if not all_data:
        print("[-] Failed to build universal dataset.")
        return
        
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dataset_{interval}.csv")
    final_df.to_csv(output_path, index=False)
    print(f"[+] Universal Dataset saved to {output_path} with shape {final_df.shape}")
    return output_path

if __name__ == "__main__":
    import sys
    interval = sys.argv[1] if len(sys.argv) > 1 else "1h"
    build_universal_dataset(interval=interval, target_pct=1.0)
