#!/usr/bin/env python3
"""
parity_check.py

Assert parity (within tolerances) between your engineered features
and reference implementations from TA-Lib and pandas-ta.

Usage examples:
    # Using a local CSV with columns: date,open,high,low,close,volume
    python parity_check.py --csv data.csv --date-col date --report parity_report.csv

    # Fetch data via yfinance (requires internet)
    python parity_check.py --ticker AAPL --start 2019-01-01 --end 2025-01-01 --report parity_report.csv

Notes:
- Only features with clear equivalents in reference libraries are checked.
- Warm-up NaNs for rolling indicators are ignored automatically.
- Tolerances are configurable.
"""

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Optional deps ---
try:
    import talib
except Exception as e:
    talib = None

try:
    import pandas_ta as pta
except Exception as e:
    pta = None

# --- Optional data fetcher ---
try:
    import yfinance as yf
except Exception:
    yf = None


def _load_ohlcv_from_csv(path, date_col=None):
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    # Normalize column names
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    return df[["open", "high", "low", "close", "volume"]]


def _load_ohlcv_from_yf(ticker, start, end):
    if yf is None:
        raise RuntimeError("yfinance not installed; install with `pip install yfinance`.")
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError("No data returned from yfinance; check ticker/dates.")
    data = data.rename(columns=str.lower)
    # yfinance uses 'adj close' sometimes; we prefer 'close'
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            raise RuntimeError(f"Downloaded data missing column: {col}")
    return data[["open", "high", "low", "close", "volume"]]


def _import_user_features(module_path):
    """
    Import your feature engineer from the provided module path.
    Expected to expose class EnhancedFinancialFeatures with method create_all_features(df).
    """
    # Allow path like "./fin_feature_preprocessing.py" or module name
    mod = None
    p = Path(module_path)
    if p.suffix == ".py" and p.exists():
        # add parent to sys.path and import by stem
        sys.path.insert(0, str(p.parent.resolve()))
        mod = importlib.import_module(p.stem)
    else:
        # try plain module import (must be importable)
        mod = importlib.import_module(module_path)

    # Class / callable
    if hasattr(mod, "EnhancedFinancialFeatures"):
        cls = getattr(mod, "EnhancedFinancialFeatures")
        return cls
    else:
        raise ImportError(
            f"Could not find `EnhancedFinancialFeatures` in {module_path}. "
            f"Please ensure your module exposes that class."
        )


def _nan_aware_stats(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return dict(n=0, max_abs_err=np.nan, mae=np.nan, rmse=np.nan)
    diff = a[mask] - b[mask]
    return dict(
        n=mask.sum(),
        max_abs_err=float(np.nanmax(np.abs(diff))),
        mae=float(np.nanmean(np.abs(diff))),
        rmse=float(np.sqrt(np.nanmean(diff**2))),
    )


def _compare_series(name, ours, ref, atol, rtol):
    # Align index
    ours, ref = ours.align(ref, join="inner")
    # Ignore warm-up NaNs: drop any rows where either is NaN
    mask = np.isfinite(ours.values) & np.isfinite(ref.values)
    if mask.sum() == 0:
        return dict(feature=name, library="(none)", n=0, passed=False,
                    max_abs_err=np.nan, mae=np.nan, rmse=np.nan, first_bad_index=None)

    a = ours.values[mask]
    b = ref.values[mask]
    stats = _nan_aware_stats(a, b)
    passed = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
    # find first failure if not passed
    first_bad_index = None
    if not passed:
        bad = np.where(np.abs(a - b) > (atol + rtol * np.abs(b)))[0]
        if len(bad) > 0:
            first_bad_index = str(ours.index[mask][bad[0]])

    return dict(
        feature=name,
        library="ref",
        n=stats["n"],
        passed=bool(passed),
        max_abs_err=stats["max_abs_err"],
        mae=stats["mae"],
        rmse=stats["rmse"],
        first_bad_index=first_bad_index,
    )


def build_reference(df, ours):
    """
    Build reference feature series using TA-Lib / pandas-ta for parity checks.
    Only features with clear equivalence are included.
    Returns a dict {feature_name: pd.Series}
    """
    refs = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"]

    # --- Momentum / Oscillators ---
    if talib is not None:
        for period in [7, 14, 21]:
            key = f"rsi_{period}"
            if key in ours:
                refs[key] = pd.Series(talib.RSI(close, timeperiod=period), index=df.index)

        if "macd" in ours or "macd_signal" in ours or "macd_hist" in ours:
            macd, sig, hist = talib.MACD(close)
            if "macd" in ours: refs["macd"] = pd.Series(macd, index=df.index)
            if "macd_signal" in ours: refs["macd_signal"] = pd.Series(sig, index=df.index)
            if "macd_hist" in ours: refs["macd_hist"] = pd.Series(hist, index=df.index)

        if "stoch_k" in ours or "stoch_d" in ours:
            slowk, slowd = talib.STOCH(high, low, close)
            if "stoch_k" in ours: refs["stoch_k"] = pd.Series(slowk, index=df.index)
            if "stoch_d" in ours: refs["stoch_d"] = pd.Series(slowd, index=df.index)

        if "williams_r" in ours:
            refs["williams_r"] = pd.Series(talib.WILLR(high, low, close), index=df.index)

        if "atr" in ours:
            refs["atr"] = pd.Series(talib.ATR(high, low, close), index=df.index)

        if "cci" in ours:
            refs["cci"] = pd.Series(talib.CCI(high, low, close), index=df.index)

        if "roc_10" in ours:
            refs["roc_10"] = pd.Series(talib.ROC(close, timeperiod=10), index=df.index)
        if "roc_20" in ours:
            refs["roc_20"] = pd.Series(talib.ROC(close, timeperiod=20), index=df.index)

    # --- Trend ---
    if talib is not None:
        if "adx" in ours:
            refs["adx"] = pd.Series(talib.ADX(high, low, close), index=df.index)
        if "adx_plus" in ours:
            refs["adx_plus"] = pd.Series(talib.PLUS_DI(high, low, close), index=df.index)
        if "adx_minus" in ours:
            refs["adx_minus"] = pd.Series(talib.MINUS_DI(high, low, close), index=df.index)

        if "sar" in ours:
            refs["sar"] = pd.Series(talib.SAR(high, low), index=df.index)

        if "aroon_up" in ours or "aroon_down" in ours:
            aroon_down, aroon_up = talib.AROON(high, low)
            if "aroon_up" in ours: refs["aroon_up"] = pd.Series(aroon_up, index=df.index)
            if "aroon_down" in ours: refs["aroon_down"] = pd.Series(aroon_down, index=df.index)

    # --- Bollinger ---
    if talib is not None:
        if "bb_upper" in ours or "bb_lower" in ours or "bb_width" in ours or "bb_percent_b" in ours or "bb_position" in ours:
            ub, mb, lb = talib.BBANDS(close, timeperiod=20)
            if "bb_upper" in ours: refs["bb_upper"] = pd.Series(ub, index=df.index)
            if "bb_lower" in ours: refs["bb_lower"] = pd.Series(lb, index=df.index)
            # Derived:
            if "bb_width" in ours:
                refs["bb_width"] = (pd.Series(ub, index=df.index) - pd.Series(lb, index=df.index)) / pd.Series(mb, index=df.index)
            if "bb_percent_b" in ours or "bb_position" in ours:
                # position / %B should be (close - lb) / (ub - lb)
                refs["bb_percent_b"] = (close - pd.Series(lb, index=df.index)) / (pd.Series(ub, index=df.index) - pd.Series(lb, index=df.index))
                if "bb_position" in ours:
                    refs["bb_position"] = refs["bb_percent_b"]

    # --- Volume-based ---
    if talib is not None:
        # OBV raw (we z-score in ours; compare raw to raw where appropriate)
        if "obv_zscore" in ours or "obv_roc" in ours:
            obv_ref = pd.Series(talib.OBV(close, vol), index=df.index)
            if "obv_roc" in ours:
                refs["obv_roc"] = obv_ref.pct_change(20)
            if "obv_zscore" in ours:
                refs["obv_zscore"] = (obv_ref - obv_ref.rolling(60).mean()) / (obv_ref.rolling(60).std())

        # AD raw
        if "ad_zscore" in ours or "ad_roc" in ours:
            ad_ref = pd.Series(talib.AD(high, low, close, vol), index=df.index)
            if "ad_roc" in ours:
                refs["ad_roc"] = ad_ref.pct_change(20)
            if "ad_zscore" in ours:
                refs["ad_zscore"] = (ad_ref - ad_ref.rolling(60).mean()) / (ad_ref.rolling(60).std())

    # CMF is not in TA-Lib; use pandas-ta if available
    if pta is not None and "cmf" in ours:
        refs["cmf"] = pta.cmf(high=high, low=low, close=close, volume=vol, length=20)

    # VWAP-20 (rolling) is not standard in TA-Lib; use our identical construction for parity
    # We create a "reference" equivalent calculation here
    if "vwap_20" in ours:
        refs["vwap_20"] = ((close * vol).rolling(20).sum()) / (vol.rolling(20).sum())

    # ATR ratio derived from ATR + price
    if "atr_ratio" in ours and talib is not None:
        atr_ref = pd.Series(talib.ATR(high, low, close), index=df.index)
        refs["atr_ratio"] = atr_ref / close

    # Distance-from-MA features (10/20/50/200) - rolling mean reference
    for per in [10, 20, 50, 200]:
        key = f"dist_from_ma{per}"
        if key in ours:
            ma = close.rolling(per).mean()
            refs[key] = (close - ma) / ma

    # Price position in daily range
    if "price_position" in ours:
        refs["price_position"] = (close - low) / (high - low)

    # Return zscore 20
    if "return_zscore_20" in ours:
        ret = close.pct_change()
        refs["return_zscore_20"] = (ret - ret.rolling(20).mean()) / (ret.rolling(20).std())

    # --- Additional features we can verify ---
    
    # Aroon oscillator (aroon_up - aroon_down)
    if "aroon_oscillator" in ours and talib is not None:
        aroon_down, aroon_up = talib.AROON(high, low)
        refs["aroon_oscillator"] = pd.Series(aroon_up - aroon_down, index=df.index)
    
    # Log returns (simple transformation of price)
    for period in [1, 2, 3, 5, 10, 20]:
        key = f"log_return_{period}d"
        if key in ours:
            refs[key] = np.log(close / close.shift(period))
    
    # Distance from 20-day high/low
    if "dist_from_20d_high" in ours:
        high_20 = high.rolling(20).max()
        refs["dist_from_20d_high"] = (close - high_20) / high_20
    
    if "dist_from_20d_low" in ours:
        low_20 = low.rolling(20).min()
        refs["dist_from_20d_low"] = (close - low_20) / low_20
    
    # SAR signal (whether price is above/below SAR)
    if "sar_signal" in ours and talib is not None:
        sar_ref = pd.Series(talib.SAR(high, low), index=df.index)
        refs["sar_signal"] = (close > sar_ref).astype(float)
    
    # Stoch K-D difference
    if "stoch_k_d_diff" in ours and talib is not None:
        slowk, slowd = talib.STOCH(high, low, close)
        refs["stoch_k_d_diff"] = pd.Series(slowk - slowd, index=df.index)
    
    # Volatility features using TA-Lib
    # Parkinson volatility (High-Low range based)
    if "volatility_parkinson_10" in ours:
        hl_ratio = np.log(high / low)
        refs["volatility_parkinson_10"] = hl_ratio.rolling(10).std() * np.sqrt(252 / (4 * np.log(2)))
    
    if "volatility_parkinson_20" in ours:
        hl_ratio = np.log(high / low)
        refs["volatility_parkinson_20"] = hl_ratio.rolling(20).std() * np.sqrt(252 / (4 * np.log(2)))
    
    # Garman-Klass volatility
    if "volatility_gk_20" in ours:
        hl = np.log(high / low) ** 2
        co = np.log(close / open_) ** 2
        refs["volatility_gk_20"] = np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(20).mean() * 252)
    
    # Yang-Zhang volatility (complex, but we can approximate with TA-Lib ATR-based approach)
    if "volatility_yz_10" in ours:
        # Simplified: use close-to-close + overnight + intraday components
        ret_cc = np.log(close / close.shift(1))
        ret_oc = np.log(open_ / close.shift(1))
        ret_co = np.log(close / open_)
        refs["volatility_yz_10"] = np.sqrt(
            ret_oc.rolling(10).var() + 0.34 * ret_co.rolling(10).var() + 0.66 * ret_cc.rolling(10).var()
        ) * np.sqrt(252)
    
    if "volatility_yz_20" in ours:
        ret_cc = np.log(close / close.shift(1))
        ret_oc = np.log(open_ / close.shift(1))
        ret_co = np.log(close / open_)
        refs["volatility_yz_20"] = np.sqrt(
            ret_oc.rolling(20).var() + 0.34 * ret_co.rolling(20).var() + 0.66 * ret_cc.rolling(20).var()
        ) * np.sqrt(252)
    
    if "volatility_yz_60" in ours:
        ret_cc = np.log(close / close.shift(1))
        ret_oc = np.log(open_ / close.shift(1))
        ret_co = np.log(close / open_)
        refs["volatility_yz_60"] = np.sqrt(
            ret_oc.rolling(60).var() + 0.34 * ret_co.rolling(60).var() + 0.66 * ret_cc.rolling(60).var()
        ) * np.sqrt(252)
    
    # Close-to-close volatility
    if "volatility_cc_20" in ours:
        ret = np.log(close / close.shift(1))
        refs["volatility_cc_20"] = ret.rolling(20).std() * np.sqrt(252)
    
    if "volatility_cc_60" in ours:
        ret = np.log(close / close.shift(1))
        refs["volatility_cc_60"] = ret.rolling(60).std() * np.sqrt(252)
    
    # Realized volatility positive/negative (semi-variance)
    if "realized_vol_positive" in ours:
        ret = close.pct_change()
        refs["realized_vol_positive"] = np.sqrt((ret[ret > 0] ** 2).rolling(20).sum() * 252)
    
    if "realized_vol_negative" in ours:
        ret = close.pct_change()
        refs["realized_vol_negative"] = np.sqrt((ret[ret < 0] ** 2).rolling(20).sum() * 252)
    
    # Volume features
    if "volume_norm" in ours:
        # Normalize by 20-day moving average
        refs["volume_norm"] = vol / vol.rolling(20).mean()
    
    if "volume_roc" in ours:
        refs["volume_roc"] = vol.pct_change(10)
    
    if "volume_zscore" in ours:
        refs["volume_zscore"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    
    if "relative_volume" in ours:
        # Volume relative to average
        refs["relative_volume"] = vol / vol.rolling(20).mean()
    
    # Dollar volume ratio
    if "dollar_volume_ma_ratio" in ours:
        dollar_vol = close * vol
        refs["dollar_volume_ma_ratio"] = dollar_vol / dollar_vol.rolling(20).mean()
    
    # Price to VWAP ratio
    if "price_vwap_ratio" in ours:
        vwap = ((close * vol).rolling(20).sum()) / (vol.rolling(20).sum())
        refs["price_vwap_ratio"] = close / vwap
    
    # Return acceleration (second derivative)
    if "return_acceleration" in ours:
        ret = close.pct_change()
        refs["return_acceleration"] = ret.diff()
    
    # Serial correlation
    if "serial_corr_5" in ours:
        ret = close.pct_change()
        refs["serial_corr_5"] = ret.rolling(20).apply(lambda x: x.autocorr(5) if len(x) > 5 else np.nan, raw=False)
    
    # Volatility of volatility
    if "vol_of_vol" in ours:
        ret = close.pct_change()
        vol_series = ret.rolling(20).std()
        refs["vol_of_vol"] = vol_series.rolling(20).std() * np.sqrt(252)
    
    # Volatility ratio (short-term vs long-term)
    if "vol_ratio_short_long" in ours:
        ret = close.pct_change()
        vol_short = ret.rolling(10).std()
        vol_long = ret.rolling(60).std()
        refs["vol_ratio_short_long"] = vol_short / vol_long
    
    # MACD divergence (if we have MACD and price divergence)
    if "macd_divergence" in ours and talib is not None:
        macd, sig, hist = talib.MACD(close)
        # Simple divergence: when price makes new high but MACD doesn't (or vice versa)
        # This is a simplified version - your implementation might be more sophisticated
        price_slope = close.diff(5) / close.shift(5)
        macd_slope = pd.Series(macd, index=df.index).diff(5) / pd.Series(macd, index=df.index).shift(5).abs()
        refs["macd_divergence"] = price_slope - macd_slope

    # ATR/ADX etc already mapped above.
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", type=str, default="fin_feature_preprocessing.py",
                    help="Path to your feature module (.py) or importable module name. "
                         "It must expose `EnhancedFinancialFeatures`.")
    ap.add_argument("--preset", type=str, default=None, help="Optional FeatureConfig preset to use.")
    ap.add_argument("--ticker", type=str, default=None, help="Ticker to fetch via yfinance.")
    ap.add_argument("--start", type=str, default="2019-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV with columns open,high,low,close,volume.")
    ap.add_argument("--date-col", type=str, default=None, help="Name of date column to set as index if using CSV.")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for parity.")
    ap.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for parity.")
    ap.add_argument("--report", type=str, default="parity_report.csv", help="Where to save the parity report CSV.")
    args = ap.parse_args()

    # --- Load data ---
    if args.csv:
        df = _load_ohlcv_from_csv(args.csv, date_col=args.date_col)
    elif args.ticker:
        df = _load_ohlcv_from_yf(args.ticker, args.start, args.end)
    else:
        raise SystemExit("Provide --csv <file> or --ticker <symbol>.")

    # --- Import user's feature class ---
    FeatureCls = _import_user_features(args.module)

    # Instantiate with preset if provided
    # EnhancedFinancialFeatures accepts feature_config, not feature_preset
    if args.preset:
        try:
            # Import FeatureConfig
            from feature_config import FeatureConfig
            config = FeatureConfig.get_preset(args.preset)
            fe = FeatureCls(feature_config=config)
        except Exception as e:
            print(f"Warning: Could not use preset '{args.preset}': {e}")
            print("Using default configuration instead.")
            fe = FeatureCls()
    else:
        fe = FeatureCls()

    # --- Build our features ---
    ours = fe.create_all_features(df)

    # --- Build references ---
    refs = build_reference(df, ours.columns)

    # --- Compare ---
    rows = []
    for name, ref_ser in refs.items():
        if name not in ours.columns:
            continue
        result = _compare_series(name, ours[name], ref_ser, atol=args.atol, rtol=args.rtol)
        result["library"] = "TA-Lib/pandas-ta/derived"
        rows.append(result)

    report = pd.DataFrame(rows).sort_values(["passed", "feature"], ascending=[True, True])
    report.to_csv(args.report, index=False)

    # Print a summary
    total = len(report)
    passed = int(report["passed"].sum()) if total else 0
    failed = total - passed
    print(f"Checked {total} features — PASSED: {passed}, FAILED: {failed}")
    if failed:
        print(report.loc[~report["passed"], ["feature", "max_abs_err", "mae", "rmse", "first_bad_index"]]
                    .to_string(index=False))

    # Show a few lines
    print("\nSample of report:")
    print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
