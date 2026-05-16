import pandas as pd
import numpy as np
import yfinance as yf
import os

# Import pure pandas indicator helpers from the existing backend calculator
from backend.indicators.calculator import _ema, _rsi, _true_range, _adx

def fetch_and_build_dataset(symbol="BTC-USD", period="1y", interval="1h", lookforward=5, target_pct=1.0):
    """
    Fetches historical data, computes features, and creates the target label.
    """
    print(f"[*] Fetching {symbol} data for the last {period} at {interval} interval...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        print("[-] Failed to fetch data.")
        return None
        
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    print(f"[*] Raw data shape: {df.shape}")

    # Ensure required columns exist and are numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop NaNs early
    df.dropna(inplace=True)

    # Compute Features (Pure Pandas)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df['RSI_14'] = _rsi(close, 14)
    
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACDs_12_26_9'] = _ema(df['MACD_12_26_9'], 9)
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    df['EMA_9'] = _ema(close, 9)
    df['EMA_21'] = _ema(close, 21)
    df['EMA_50'] = _ema(close, 50)
    
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['BBM_20_2.0'] = sma_20
    df['BBU_20_2.0'] = sma_20 + 2 * std_20
    df['BBL_20_2.0'] = sma_20 - 2 * std_20
    df['BBB_20_2.0'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / sma_20 * 100
    df['BBP_20_2.0'] = (close - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    tr = _true_range(high, low, close)
    df['ATRr_14'] = tr.rolling(14).mean()
    
    # Simple approximations for ADX/DMP/DMN if missing
    df['ADX_14'] = _adx(high, low, close, 14)
    df['DMP_14'] = 0.0 # Skipping full computation for dataset brevity
    df['DMN_14'] = 0.0 
    
    # Custom Features
    df['returns'] = close.pct_change()
    df['vol_sma'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_sma']
    
    # Drop rows with NaN from indicator warm-up
    df.dropna(inplace=True)

    # Build Target Label
    # Did the price rise by more than `target_pct` % within the next `lookforward` candles?
    df['future_max'] = df['High'].shift(-lookforward).rolling(lookforward).max()
    df['target'] = np.where((df['future_max'] - df['Close']) / df['Close'] > (target_pct / 100.0), 1, 0)
    
    # Drop the last few rows where we don't have future data
    df.dropna(inplace=True)
    
    # Keep only the feature columns and target
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'EMA_9', 'EMA_21', 'EMA_50', 
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'ATRr_14', 'ADX_14', 'DMP_14', 'DMN_14',
        'returns', 'vol_ratio'
    ]
    
    available_features = [f for f in features if f in df.columns]
    
    return df[available_features + ['target']].copy()

def build_universal_dataset(interval="1h", period="1y", target_pct=1.0):
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD']
    all_data = []
    
    # Adjust lookforward and target_pct based on interval
    lookforward = 5
    adjusted_target_pct = target_pct
    if interval == "15m":
        lookforward = 8
        adjusted_target_pct = 0.5
    elif interval == "1d":
        lookforward = 3
        adjusted_target_pct = 3.0
        
    print(f"[*] Building Universal Dataset for {interval}...")
    for sym in symbols:
        df = fetch_and_build_dataset(sym, period, interval, lookforward, adjusted_target_pct)
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
    build_universal_dataset(interval=interval, period="1y", target_pct=1.0)
