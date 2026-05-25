import os
import joblib
import pandas as pd
import numpy as np

class MLPredictor:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(models_dir):
            print("[-] ML Models directory not found.")
            return
            
        for file in os.listdir(models_dir):
            if file.startswith("rf_model_") and file.endswith(".joblib"):
                interval = file.replace("rf_model_", "").replace(".joblib", "")
                model_path = os.path.join(models_dir, file)
                try:
                    data = joblib.load(model_path)
                    self.models[interval] = {
                        "model": data.get("model"),
                        "features": data.get("features", [])
                    }
                    print(f"[*] ML Predictor loaded for {interval}.")
                except Exception as e:
                    print(f"[-] Failed to load ML model for {interval}: {e}")
                    
        # Backward compatibility for the old generic model
        old_path = os.path.join(models_dir, "rf_model.joblib")
        if os.path.exists(old_path) and "1h" not in self.models:
            try:
                data = joblib.load(old_path)
                self.models["1h"] = {"model": data.get("model"), "features": data.get("features", [])}
                print("[*] ML Predictor loaded for 1h (legacy).")
            except Exception: pass

    def predict(self, indicators: dict, candles: list[dict], interval: str = "1h") -> dict:
        """
        Takes indicators and historical candles list, builds a scale-invariant feature space
        with momentum trajectory vectors (lags, velocity, acceleration), and returns prediction probability.
        """
        model_data = self.models.get(interval)
        
        if not candles or len(candles) < 3:
            return {"up_prob": 50.0, "down_prob": 50.0, "status": "insufficient_data"}

        if not model_data or not model_data["model"] or not model_data["features"]:
            # --- HEURISTIC FALLBACK ---
            rsi = self._get_last(indicators.get('rsi'))
            macd = self._get_last(indicators.get('macd_histogram'))
            
            up_prob = 50.0
            if rsi < 30: up_prob += 15
            elif rsi > 70: up_prob -= 15
            if macd > 0: up_prob += 10
            elif macd < 0: up_prob -= 10
            
            up_prob += np.random.uniform(-3, 3)
            up_prob = min(99.0, max(1.0, up_prob))
            
            return {
                "up_prob": round(up_prob, 1),
                "down_prob": round(100.0 - up_prob, 1),
                "status": "ok"
            }

        # Helper to construct scale-invariant features at a dynamic historical step (1=now, 2=lag1, 3=lag2)
        def get_feature_at_step(n):
            candle = candles[-n]
            o_val = candle.get('open', 0.0) or candle.get('Open', 0.0)
            h_val = candle.get('high', 0.0) or candle.get('High', 0.0)
            l_val = candle.get('low', 0.0) or candle.get('Low', 0.0)
            c_val = candle.get('close', 0.0) or candle.get('Close', 0.0)
            
            rsi = self._get_nth_last(indicators.get('rsi'), n)
            
            macd_hist = self._get_nth_last(indicators.get('macd_histogram'), n)
            macd_ratio = (macd_hist / c_val * 100) if c_val > 0 else 0.0
            
            ema9 = self._get_nth_last(indicators.get('ema_9'), n)
            ema9_ratio = ((c_val - ema9) / ema9 * 100) if ema9 > 0 else 0.0
            
            ema21 = self._get_nth_last(indicators.get('ema_21'), n)
            ema21_ratio = ((c_val - ema21) / ema21 * 100) if ema21 > 0 else 0.0
            
            ema50 = self._get_nth_last(indicators.get('ema_50'), n)
            ema50_ratio = ((c_val - ema50) / ema50 * 100) if ema50 > 0 else 0.0
            
            bbb = self._get_nth_last(indicators.get('bb_bandwidth'), n)
            
            bbu = self._get_nth_last(indicators.get('bb_upper'), n)
            bbl = self._get_nth_last(indicators.get('bb_lower'), n)
            bbp = ((c_val - bbl) / (bbu - bbl)) if (bbu - bbl) > 0 else 0.5
            
            atr = self._get_nth_last(indicators.get('atr'), n)
            atr_ratio = (atr / c_val * 100) if c_val > 0 else 0.0
            
            adx = self._get_nth_last(indicators.get('adx'), n)
            
            returns = ((c_val - o_val) / o_val * 100) if o_val > 0 else 0.0
            spread_pct = ((h_val - l_val) / c_val * 100) if c_val > 0 else 0.0
            vol_ratio = self._get_nth_last(indicators.get('volume_ratio'), n)
            
            return {
                'RSI_14': rsi,
                'MACDh_ratio': macd_ratio,
                'EMA9_ratio': ema9_ratio,
                'EMA21_ratio': ema21_ratio,
                'EMA50_ratio': ema50_ratio,
                'BBB_20_2.0': bbb,
                'BBP_20_2.0': bbp,
                'ATRr_ratio': atr_ratio,
                'ADX_14': adx,
                'returns': returns,
                'spread_pct': spread_pct,
                'vol_ratio': vol_ratio
            }

        try:
            # Query states at t0, t-1, t-2
            t0 = get_feature_at_step(1)
            t1 = get_feature_at_step(2)
            t2 = get_feature_at_step(3)
            
            # Construct row
            row = {}
            for k, v in t0.items():
                row[k] = v
                
            # Trajectories matching dataset builder
            trajectory_cols = ['RSI_14', 'MACDh_ratio', 'BBP_20_2.0', 'returns', 'vol_ratio']
            for col in trajectory_cols:
                row[f'{col}_lag1'] = t1[col]
                row[f'{col}_lag2'] = t2[col]
                row[f'{col}_velocity'] = t0[col] - t1[col]
                row[f'{col}_acceleration'] = (t0[col] - t1[col]) - (t1[col] - t2[col])
            
            # Match only target trained features in correct order
            final_row = {feat: row.get(feat, 0.0) for feat in model_data["features"]}
            
            df_row = pd.DataFrame([final_row])
            df_row.fillna(0, inplace=True)

            # Predict probability
            probs = model_data["model"].predict_proba(df_row)[0]
            down_prob = probs[0] * 100
            up_prob = probs[1] * 100
            return {
                "up_prob": round(up_prob, 1), 
                "down_prob": round(down_prob, 1),
                "status": "ok"
            }
        except Exception as e:
            return {"up_prob": 50.0, "down_prob": 50.0, "status": f"error: {str(e)}"}

    def _get_last(self, ind_list):
        if ind_list and isinstance(ind_list, list) and len(ind_list) > 0:
            return ind_list[-1].get('value', 0.0)
        return 0.0

    def _get_nth_last(self, ind_list, n=1):
        if ind_list and isinstance(ind_list, list) and len(ind_list) >= n:
            return ind_list[-n].get('value', 0.0)
        return 0.0

predictor = MLPredictor()
