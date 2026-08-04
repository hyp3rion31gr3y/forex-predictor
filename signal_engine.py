import math
from typing import Any, Optional

import pandas as pd
import ta

from data_fetcher import fetch_forex_data


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _try_indicator(df: pd.DataFrame, col: str, func):
    """Safely compute an indicator column; fill with NaN on failure."""
    try:
        df[col] = func()
    except Exception:
        df[col] = float("nan")


def compute_indicators(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Add all technical indicator columns to the DataFrame.

    Uses shorter indicator windows for intraday intervals so that
    the signals react faster and remain meaningful on sub-daily bars.
    """
    _intra = interval not in ("1d", "1wk", "1mo")
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI — window 14 for both (well-established)
    _try_indicator(df, "RSI", lambda: ta.momentum.RSIIndicator(close, window=14).rsi())

    # MACD — faster (8,17,9) for intraday; standard (12,26,9) for daily
    try:
        mf, ms, msig = (8, 17, 9) if _intra else (12, 26, 9)
        macd = ta.trend.MACD(close, window_slow=ms, window_fast=mf, window_sign=msig)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
    except Exception:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = float("nan")

    # SMAs — keep same windows; signal function skips SMA_200 for intraday
    _try_indicator(df, "SMA_20", lambda: ta.trend.SMAIndicator(close, window=20).sma_indicator())
    _try_indicator(df, "SMA_50", lambda: ta.trend.SMAIndicator(close, window=50).sma_indicator())
    _try_indicator(df, "SMA_200", lambda: ta.trend.SMAIndicator(close, window=200).sma_indicator())

    # EMAs
    _try_indicator(df, "EMA_12", lambda: ta.trend.EMAIndicator(close, window=12).ema_indicator())
    _try_indicator(df, "EMA_26", lambda: ta.trend.EMAIndicator(close, window=26).ema_indicator())
    _try_indicator(df, "EMA_9", lambda: ta.trend.EMAIndicator(close, window=9).ema_indicator())
    _try_indicator(df, "EMA_21", lambda: ta.trend.EMAIndicator(close, window=21).ema_indicator())

    # Bollinger Bands — tighter window=10 for intraday
    try:
        bb_win = 10 if _intra else 20
        bb = ta.volatility.BollingerBands(close, window=bb_win, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()
    except Exception:
        df["BB_Upper"] = df["BB_Middle"] = df["BB_Lower"] = float("nan")

    # Stochastic — faster (5,3) for intraday; standard (14,3) for daily
    try:
        stoch_win = 5 if _intra else 14
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=stoch_win, smooth_window=3)
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
    except Exception:
        df["Stoch_K"] = df["Stoch_D"] = float("nan")

    # ADX — shorter window=10 for intraday
    try:
        adx_win = 10 if _intra else 14
        adx = ta.trend.ADXIndicator(high, low, close, window=adx_win)
        df["ADX"] = adx.adx()
        df["ADX_Pos"] = adx.adx_pos()
        df["ADX_Neg"] = adx.adx_neg()
    except Exception:
        df["ADX"] = df["ADX_Pos"] = df["ADX_Neg"] = float("nan")

    # ATR — shorter window=10 for intraday
    atr_win = 10 if _intra else 14
    _try_indicator(df, "ATR", lambda: ta.volatility.AverageTrueRange(high, low, close, window=atr_win).average_true_range())

    # Volume-dependent indicators
    has_volume = "Volume" in df.columns and df["Volume"].sum() > 0

    if has_volume:
        volume = df["Volume"]
        # MFI (14) — Money Flow Index
        _try_indicator(df, "MFI", lambda: ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index())
        # OBV — On-Balance Volume
        _try_indicator(df, "OBV", lambda: ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume())
        # VWAP — Volume Weighted Average Price (daily reset)
        try:
            typical_price = (high + low + close) / 3
            dates = df.index.date
            cum_tp_vol = (typical_price * volume).groupby(dates).cumsum()
            cum_vol = volume.groupby(dates).cumsum()
            df["VWAP"] = cum_tp_vol / cum_vol.replace(0, float("nan"))
        except Exception:
            df["VWAP"] = float("nan")

    # Parabolic SAR — smaller step for intraday to reduce whipsaws
    try:
        psar_step = 0.01 if _intra else 0.02
        psar = ta.trend.PSARIndicator(high, low, close, step=psar_step, max_step=0.2)
        df["PSAR"] = psar.psar()
        df["PSAR_Up"] = psar.psar_up()
        df["PSAR_Down"] = psar.psar_down()
    except Exception:
        df["PSAR"] = df["PSAR_Up"] = df["PSAR_Down"] = float("nan")

    # CCI — shorter window=10 for intraday
    cci_win = 10 if _intra else 20
    _try_indicator(df, "CCI", lambda: ta.trend.CCIIndicator(high, low, close, window=cci_win).cci())

    return df


# ---------------------------------------------------------------------------
# Individual signal functions  (return score -2..+2 and explanation)
# ---------------------------------------------------------------------------

def _safe(val: Any) -> bool:
    """Return True if value is a usable finite number."""
    if val is None:
        return False
    try:
        return not (math.isnan(val) or math.isinf(val))
    except (TypeError, ValueError):
        return False


# --- Change 3: RSI with trend-context suppression ---
def _rsi_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    val = df["RSI"].iloc[-1]
    if not _safe(val):
        return None
    val = round(float(val), 2)

    # Tighter thresholds for intraday (RSI rarely hits 30/70 on short bars)
    if is_intraday:
        if val < 25:
            score, label, expl = 2, "Strong Buy", f"RSI {val} — deeply oversold"
        elif val < 35:
            score, label, expl = 1, "Buy", f"RSI {val} — oversold"
        elif val > 75:
            score, label, expl = -2, "Strong Sell", f"RSI {val} — deeply overbought"
        elif val > 65:
            score, label, expl = -1, "Sell", f"RSI {val} — overbought"
        elif val > 55:
            score, label, expl = -0.5, "Slightly Bearish", f"RSI {val} — leaning overbought"
        elif val < 45:
            score, label, expl = 0.5, "Slightly Bullish", f"RSI {val} — leaning oversold"
        else:
            score, label, expl = 0, "Neutral", f"RSI {val} — neutral zone"
    else:
        if val < 20:
            score, label, expl = 2, "Strong Buy", f"RSI {val} — deeply oversold"
        elif val < 30:
            score, label, expl = 1, "Buy", f"RSI {val} — oversold"
        elif val > 80:
            score, label, expl = -2, "Strong Sell", f"RSI {val} — deeply overbought"
        elif val > 70:
            score, label, expl = -1, "Sell", f"RSI {val} — overbought"
        elif val > 60:
            score, label, expl = -0.5, "Slightly Bearish", f"RSI {val} — leaning overbought"
        elif val < 40:
            score, label, expl = 0.5, "Slightly Bullish", f"RSI {val} — leaning oversold"
        else:
            score, label, expl = 0, "Neutral", f"RSI {val} — neutral zone"

    # Dampen counter-trend RSI signals when ADX > 30 with clear DI direction
    if "ADX" in df.columns and "ADX_Pos" in df.columns and "ADX_Neg" in df.columns:
        adx_val = df["ADX"].iloc[-1]
        plus_di = df["ADX_Pos"].iloc[-1]
        minus_di = df["ADX_Neg"].iloc[-1]
        if _safe(adx_val) and _safe(plus_di) and _safe(minus_di):
            adx_val, plus_di, minus_di = float(adx_val), float(plus_di), float(minus_di)
            if adx_val > 30:
                # Strong uptrend: dampen overbought (sell) signals
                if plus_di > minus_di and score < 0:
                    score = round(score * 0.3, 2)
                    expl += " (dampened: strong uptrend per ADX)"
                    label = "Slightly Bearish" if score < 0 else "Neutral"
                # Strong downtrend: dampen oversold (buy) signals
                elif minus_di > plus_di and score > 0:
                    score = round(score * 0.3, 2)
                    expl += " (dampened: strong downtrend per ADX)"
                    label = "Slightly Bullish" if score > 0 else "Neutral"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "RSI (14)", "value": val, "score": score, "signal": label, "explanation": expl, "bet": bet}


def _macd_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    macd_val = df["MACD"].iloc[-1]
    sig_val = df["MACD_Signal"].iloc[-1]
    hist = df["MACD_Hist"].iloc[-1]
    if not (_safe(macd_val) and _safe(sig_val) and _safe(hist)):
        return None
    macd_val, sig_val, hist = float(macd_val), float(sig_val), float(hist)

    # Check for crossover in last 3 bars
    recent_hist = df["MACD_Hist"].dropna().tail(3)
    crossover = False
    cross_dir = 0
    if len(recent_hist) >= 2:
        vals = recent_hist.values
        for i in range(1, len(vals)):
            if _safe(vals[i]) and _safe(vals[i - 1]):
                if vals[i] > 0 and vals[i - 1] <= 0:
                    crossover, cross_dir = True, 1
                elif vals[i] < 0 and vals[i - 1] >= 0:
                    crossover, cross_dir = True, -1

    if crossover and cross_dir > 0:
        score, label = 2, "Strong Buy"
        expl = "MACD crossed above signal line — bullish crossover"
    elif crossover and cross_dir < 0:
        score, label = -2, "Strong Sell"
        expl = "MACD crossed below signal line — bearish crossover"
    elif hist > 0:
        score = min(2, hist / max(abs(sig_val), 0.0001))
        label = "Strong Buy" if score >= 1.5 else "Buy"
        expl = f"MACD ({macd_val:.4f}) above signal ({sig_val:.4f})"
    else:
        score = max(-2, hist / max(abs(sig_val), 0.0001))
        label = "Strong Sell" if score <= -1.5 else "Sell"
        expl = f"MACD ({macd_val:.4f}) below signal ({sig_val:.4f})"

    score = round(score, 2)
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    macd_name = "MACD (8,17,9)" if is_intraday else "MACD (12,26,9)"
    return {"name": macd_name, "value": round(macd_val, 5), "score": score, "signal": label, "explanation": expl, "bet": bet}


def _sma_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    close = float(df["Close"].iloc[-1])
    sma20 = df["SMA_20"].iloc[-1]
    sma50 = df["SMA_50"].iloc[-1]

    # Skip SMA_200 for intraday — it covers only hours, not months
    sma_list = [("SMA20", sma20), ("SMA50", sma50)]
    if not is_intraday:
        sma200 = df["SMA_200"].iloc[-1]
        sma_list.append(("SMA200", sma200))

    parts = []
    score = 0
    count = 0

    for label, val in sma_list:
        if _safe(val):
            val = float(val)
            count += 1
            if close > val:
                score += 1
                parts.append(f"Price above {label}")
            else:
                score -= 1
                parts.append(f"Price below {label}")

    if count == 0:
        return None

    score = score / count  # normalise to -1..+1
    # Amplify if all agree
    if abs(score) == 1:
        score *= 1.5

    if score > 0:
        signal_label = "Buy" if score >= 1 else "Slightly Bullish"
    elif score < 0:
        signal_label = "Sell" if score <= -1 else "Slightly Bearish"
    else:
        signal_label = "Neutral"

    score = round(score, 2)
    sma_name = "SMA (20/50)" if is_intraday else "SMA (20/50/200)"
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {
        "name": sma_name,
        "value": f"{close:.4f}",
        "score": score,
        "signal": signal_label,
        "explanation": "; ".join(parts),
        "bet": bet,
    }


def _ema_signal(df: pd.DataFrame) -> Optional[dict]:
    ema12 = df["EMA_12"].iloc[-1]
    ema26 = df["EMA_26"].iloc[-1]
    if not (_safe(ema12) and _safe(ema26)):
        return None
    ema12, ema26 = float(ema12), float(ema26)
    close = float(df["Close"].iloc[-1])

    diff_pct = (ema12 - ema26) / ema26 * 100

    if ema12 > ema26 and close > ema12:
        score, label = 1.5, "Buy"
        expl = f"EMA12 ({ema12:.4f}) > EMA26 ({ema26:.4f}), price above both"
    elif ema12 > ema26:
        score, label = 0.5, "Slightly Bullish"
        expl = f"EMA12 ({ema12:.4f}) > EMA26 ({ema26:.4f})"
    elif ema12 < ema26 and close < ema12:
        score, label = -1.5, "Sell"
        expl = f"EMA12 ({ema12:.4f}) < EMA26 ({ema26:.4f}), price below both"
    elif ema12 < ema26:
        score, label = -0.5, "Slightly Bearish"
        expl = f"EMA12 ({ema12:.4f}) < EMA26 ({ema26:.4f})"
    else:
        score, label = 0, "Neutral"
        expl = "EMA12 ≈ EMA26"

    score = round(score, 2)
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "EMA (12/26)", "value": round(diff_pct, 3), "score": score, "signal": label, "explanation": expl, "bet": bet}


def _bollinger_signal(df: pd.DataFrame) -> Optional[dict]:
    close = float(df["Close"].iloc[-1])
    upper = df["BB_Upper"].iloc[-1]
    lower = df["BB_Lower"].iloc[-1]
    middle = df["BB_Middle"].iloc[-1]
    if not (_safe(upper) and _safe(lower) and _safe(middle)):
        return None
    upper, lower, middle = float(upper), float(lower), float(middle)
    band_width = upper - lower
    if band_width == 0:
        return None

    position = (close - lower) / band_width  # 0 = at lower, 1 = at upper

    if position <= 0.05:
        score, label = 2, "Strong Buy"
        expl = f"Price at/below lower band ({lower:.4f}) — potential reversal up"
    elif position <= 0.2:
        score, label = 1, "Buy"
        expl = f"Price near lower band — oversold zone"
    elif position >= 0.95:
        score, label = -2, "Strong Sell"
        expl = f"Price at/above upper band ({upper:.4f}) — potential reversal down"
    elif position >= 0.8:
        score, label = -1, "Sell"
        expl = f"Price near upper band — overbought zone"
    else:
        score, label = 0, "Neutral"
        expl = f"Price within bands (position {position:.0%})"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "Bollinger Bands", "value": round(position * 100, 1), "score": score, "signal": label, "explanation": expl, "bet": bet}


# --- Change 2: Stochastic with K/D crossover confirmation ---
def _stochastic_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    k = df["Stoch_K"].iloc[-1]
    d = df["Stoch_D"].iloc[-1]
    if not (_safe(k) and _safe(d)):
        return None
    k, d = float(k), float(d)

    # Check for K/D crossover in last 3 bars
    k_series = df["Stoch_K"].dropna().tail(3)
    d_series = df["Stoch_D"].dropna().tail(3)
    has_bullish_cross = False
    has_bearish_cross = False
    if len(k_series) >= 2 and len(d_series) >= 2:
        common_idx = k_series.index.intersection(d_series.index)
        if len(common_idx) >= 2:
            k_vals = k_series.loc[common_idx].values
            d_vals = d_series.loc[common_idx].values
            for i in range(1, len(k_vals)):
                if _safe(k_vals[i]) and _safe(d_vals[i]) and _safe(k_vals[i - 1]) and _safe(d_vals[i - 1]):
                    if k_vals[i] > d_vals[i] and k_vals[i - 1] <= d_vals[i - 1]:
                        has_bullish_cross = True
                    elif k_vals[i] < d_vals[i] and k_vals[i - 1] >= d_vals[i - 1]:
                        has_bearish_cross = True

    if k < 20 and d < 20:
        base_score = 2 if k < 10 else 1
        if has_bullish_cross:
            score = base_score
            label = "Strong Buy" if score == 2 else "Buy"
            expl = f"%K={k:.1f}, %D={d:.1f} — oversold with K/D bullish crossover"
        else:
            score = 0.5
            label = "Slightly Bullish"
            expl = f"%K={k:.1f}, %D={d:.1f} — oversold but no K/D crossover yet"
    elif k > 80 and d > 80:
        base_score = -2 if k > 90 else -1
        if has_bearish_cross:
            score = base_score
            label = "Strong Sell" if score == -2 else "Sell"
            expl = f"%K={k:.1f}, %D={d:.1f} — overbought with K/D bearish crossover"
        else:
            score = -0.5
            label = "Slightly Bearish"
            expl = f"%K={k:.1f}, %D={d:.1f} — overbought but no K/D crossover yet"
    elif k > d:
        score, label = 0.5, "Slightly Bullish"
        expl = f"%K={k:.1f} > %D={d:.1f} — bullish momentum"
    elif k < d:
        score, label = -0.5, "Slightly Bearish"
        expl = f"%K={k:.1f} < %D={d:.1f} — bearish momentum"
    else:
        score, label = 0, "Neutral"
        expl = f"%K={k:.1f} ≈ %D={d:.1f}"

    stoch_name = "Stochastic (5,3)" if is_intraday else "Stochastic (14,3)"
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": stoch_name, "value": round(k, 2), "score": score, "signal": label, "explanation": expl, "bet": bet}


def _adx_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    adx_val = df["ADX"].iloc[-1]
    plus_di = df["ADX_Pos"].iloc[-1]
    minus_di = df["ADX_Neg"].iloc[-1]
    if not (_safe(adx_val) and _safe(plus_di) and _safe(minus_di)):
        return None
    adx_val, plus_di, minus_di = float(adx_val), float(plus_di), float(minus_di)

    # Lower trending threshold for intraday (ADX(10) runs lower)
    trend_threshold = 20 if is_intraday else 25
    trending = adx_val > trend_threshold

    if plus_di > minus_di:
        if trending:
            score, label = 1.5, "Buy"
            expl = f"ADX {adx_val:.1f} (strong trend), +DI ({plus_di:.1f}) > -DI ({minus_di:.1f}) — bullish trend"
        else:
            score, label = 0.5, "Slightly Bullish"
            expl = f"ADX {adx_val:.1f} (weak trend), +DI > -DI — mild bullish"
    elif minus_di > plus_di:
        if trending:
            score, label = -1.5, "Sell"
            expl = f"ADX {adx_val:.1f} (strong trend), -DI ({minus_di:.1f}) > +DI ({plus_di:.1f}) — bearish trend"
        else:
            score, label = -0.5, "Slightly Bearish"
            expl = f"ADX {adx_val:.1f} (weak trend), -DI > +DI — mild bearish"
    else:
        score, label = 0, "Neutral"
        expl = f"ADX {adx_val:.1f}, +DI ≈ -DI — no directional bias"

    adx_name = "ADX (10)" if is_intraday else "ADX (14)"
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": adx_name, "value": round(adx_val, 2), "score": score, "signal": label, "explanation": expl, "bet": bet}


def _mfi_signal(df: pd.DataFrame) -> Optional[dict]:
    if "MFI" not in df.columns:
        return None
    val = df["MFI"].iloc[-1]
    if not _safe(val):
        return None
    val = round(float(val), 2)
    if val < 10:
        score, label, expl = 2, "Strong Buy", f"MFI {val} — deeply oversold (volume-confirmed)"
    elif val < 20:
        score, label, expl = 1, "Buy", f"MFI {val} — oversold (volume-confirmed)"
    elif val > 90:
        score, label, expl = -2, "Strong Sell", f"MFI {val} — deeply overbought (volume-confirmed)"
    elif val > 80:
        score, label, expl = -1, "Sell", f"MFI {val} — overbought (volume-confirmed)"
    elif val > 65:
        score, label, expl = -0.5, "Slightly Bearish", f"MFI {val} — leaning overbought"
    elif val < 35:
        score, label, expl = 0.5, "Slightly Bullish", f"MFI {val} — leaning oversold"
    else:
        score, label, expl = 0, "Neutral", f"MFI {val} — neutral zone"
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "MFI (14)", "value": val, "score": score, "signal": label, "explanation": expl, "bet": bet}


# --- Change 1: OBV with EMA smoothing and 5% threshold ---
def _obv_signal(df: pd.DataFrame) -> Optional[dict]:
    if "OBV" not in df.columns:
        return None
    obv = df["OBV"].dropna()
    if len(obv) < 20:
        return None

    # Smooth OBV with 10-period EMA to reduce single-spike noise
    obv_ema = obv.ewm(span=10, adjust=False).mean()

    obv_ema_recent = obv_ema.tail(20)
    close_recent = df["Close"].tail(20)

    obv_change = float(obv_ema_recent.iloc[-1] - obv_ema_recent.iloc[0])
    price_change = float(close_recent.iloc[-1] - close_recent.iloc[0])

    obv_val = float(obv.iloc[-1])

    # 5% minimum threshold — OBV change must be meaningful relative to its mean
    obv_mean = float(obv_ema.abs().mean())
    if obv_mean > 0 and abs(obv_change) / obv_mean < 0.05:
        score, label = 0, "Neutral"
        expl = "EMA-smoothed OBV change below 5% threshold — no clear volume signal"
    elif price_change > 0 and obv_change < 0:
        score, label = -1, "Sell"
        expl = "Price rising but smoothed OBV falling — weak rally, distribution"
    elif price_change < 0 and obv_change > 0:
        score, label = 1, "Buy"
        expl = "Price falling but smoothed OBV rising — stealth accumulation"
    elif price_change > 0 and obv_change > 0:
        score, label = 0.5, "Slightly Bullish"
        expl = "Price and smoothed OBV both rising — volume confirms uptrend"
    elif price_change < 0 and obv_change < 0:
        score, label = -0.5, "Slightly Bearish"
        expl = "Price and smoothed OBV both falling — volume confirms downtrend"
    else:
        score, label = 0, "Neutral"
        expl = "Smoothed OBV flat — no clear volume trend"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    # Display OBV in millions/thousands for readability
    if abs(obv_val) >= 1e6:
        display_val = f"{obv_val/1e6:.1f}M"
    elif abs(obv_val) >= 1e3:
        display_val = f"{obv_val/1e3:.1f}K"
    else:
        display_val = f"{obv_val:.0f}"
    return {"name": "OBV", "value": display_val, "score": score, "signal": label, "explanation": expl, "bet": bet}


def _rsi_divergence_signal(df: pd.DataFrame) -> Optional[dict]:
    """Detect bullish/bearish RSI divergence via peak/trough detection (30-bar lookback)."""
    if "RSI" not in df.columns:
        return None
    rsi = df["RSI"].dropna()
    close = df["Close"].dropna()
    if len(rsi) < 30 or len(close) < 30:
        return None

    lookback = 30
    rsi_window = rsi.tail(lookback).values
    close_window = close.tail(lookback).values

    # Find local troughs (for bullish divergence) and peaks (for bearish divergence)
    # A trough is a point lower than its neighbours; a peak is higher
    troughs = []
    peaks = []
    for i in range(2, len(rsi_window) - 2):
        # Trough
        if (rsi_window[i] < rsi_window[i-1] and rsi_window[i] < rsi_window[i-2] and
                rsi_window[i] < rsi_window[i+1] and rsi_window[i] < rsi_window[i+2]):
            troughs.append(i)
        # Peak
        if (rsi_window[i] > rsi_window[i-1] and rsi_window[i] > rsi_window[i-2] and
                rsi_window[i] > rsi_window[i+1] and rsi_window[i] > rsi_window[i+2]):
            peaks.append(i)

    # Bullish divergence: price makes lower low, RSI makes higher low
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if close_window[t2] < close_window[t1] and rsi_window[t2] > rsi_window[t1]:
            score = 1.5
            label = "Buy"
            expl = "Bullish RSI divergence — price lower low but RSI higher low (reversal warning)"
            bet = "UP"
            return {"name": "RSI Divergence", "value": round(float(rsi_window[t2]), 2),
                    "score": score, "signal": label, "explanation": expl, "bet": bet}

    # Bearish divergence: price makes higher high, RSI makes lower high
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if close_window[p2] > close_window[p1] and rsi_window[p2] < rsi_window[p1]:
            score = -1.5
            label = "Sell"
            expl = "Bearish RSI divergence — price higher high but RSI lower high (reversal warning)"
            bet = "DOWN"
            return {"name": "RSI Divergence", "value": round(float(rsi_window[p2]), 2),
                    "score": score, "signal": label, "explanation": expl, "bet": bet}

    # No divergence found
    return None


# --- Change 7: MACD divergence detection ---
def _macd_divergence_signal(df: pd.DataFrame) -> Optional[dict]:
    """Detect bullish/bearish MACD histogram divergence (30-bar lookback)."""
    if "MACD_Hist" not in df.columns:
        return None
    hist = df["MACD_Hist"].dropna()
    close = df["Close"].dropna()
    if len(hist) < 30 or len(close) < 30:
        return None

    lookback = 30
    hist_window = hist.tail(lookback).values
    close_window = close.tail(lookback).values

    troughs = []
    peaks = []
    for i in range(2, len(hist_window) - 2):
        if (hist_window[i] < hist_window[i - 1] and hist_window[i] < hist_window[i - 2] and
                hist_window[i] < hist_window[i + 1] and hist_window[i] < hist_window[i + 2]):
            troughs.append(i)
        if (hist_window[i] > hist_window[i - 1] and hist_window[i] > hist_window[i - 2] and
                hist_window[i] > hist_window[i + 1] and hist_window[i] > hist_window[i + 2]):
            peaks.append(i)

    # Bullish divergence: price lower low, histogram higher low
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if close_window[t2] < close_window[t1] and hist_window[t2] > hist_window[t1]:
            return {"name": "MACD Divergence", "value": round(float(hist_window[t2]), 5),
                    "score": 1.5, "signal": "Buy",
                    "explanation": "Bullish MACD divergence — price lower low but histogram higher low (reversal warning)",
                    "bet": "UP"}

    # Bearish divergence: price higher high, histogram lower high
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if close_window[p2] > close_window[p1] and hist_window[p2] < hist_window[p1]:
            return {"name": "MACD Divergence", "value": round(float(hist_window[p2]), 5),
                    "score": -1.5, "signal": "Sell",
                    "explanation": "Bearish MACD divergence — price higher high but histogram lower high (reversal warning)",
                    "bet": "DOWN"}

    return None


def _psar_signal(df: pd.DataFrame) -> Optional[dict]:
    if "PSAR" not in df.columns:
        return None
    psar_val = df["PSAR"].iloc[-1]
    close_val = float(df["Close"].iloc[-1])
    if not _safe(psar_val):
        return None
    psar_val = float(psar_val)

    # Check for flip (SAR changed sides) in last 3 bars
    psar_series = df["PSAR"].dropna().tail(3)
    close_series = df["Close"].tail(3)
    flip = False
    if len(psar_series) >= 2 and len(close_series) >= 2:
        prev_above = float(psar_series.iloc[-2]) > float(close_series.iloc[-2])
        curr_above = psar_val > close_val
        if prev_above != curr_above:
            flip = True

    if close_val > psar_val:
        if flip:
            score, label = 1.5, "Buy"
            expl = f"SAR flipped bullish — reversal signal (SAR: {psar_val:.4f})"
        else:
            score, label = 0.5, "Slightly Bullish"
            expl = f"Price above SAR ({psar_val:.4f}) — uptrend"
    else:
        if flip:
            score, label = -1.5, "Sell"
            expl = f"SAR flipped bearish — reversal signal (SAR: {psar_val:.4f})"
        else:
            score, label = -0.5, "Slightly Bearish"
            expl = f"Price below SAR ({psar_val:.4f}) — downtrend"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "Parabolic SAR", "value": round(psar_val, 5), "score": score, "signal": label, "explanation": expl, "bet": bet}


def _cci_signal(df: pd.DataFrame, is_intraday: bool = False) -> Optional[dict]:
    if "CCI" not in df.columns:
        return None
    val = df["CCI"].iloc[-1]
    if not _safe(val):
        return None
    val = round(float(val), 2)

    # Tighter thresholds for intraday (CCI(10) on short bars stays closer to zero)
    if is_intraday:
        if val < -150:
            score, label, expl = 2, "Strong Buy", f"CCI {val} — extreme oversold"
        elif val < -75:
            score, label, expl = 1, "Buy", f"CCI {val} — oversold"
        elif val > 150:
            score, label, expl = -2, "Strong Sell", f"CCI {val} — extreme overbought"
        elif val > 75:
            score, label, expl = -1, "Sell", f"CCI {val} — overbought"
        elif val > 30:
            score, label, expl = -0.5, "Slightly Bearish", f"CCI {val} — leaning overbought"
        elif val < -30:
            score, label, expl = 0.5, "Slightly Bullish", f"CCI {val} — leaning oversold"
        else:
            score, label, expl = 0, "Neutral", f"CCI {val} — neutral zone"
    else:
        if val < -200:
            score, label, expl = 2, "Strong Buy", f"CCI {val} — extreme oversold"
        elif val < -100:
            score, label, expl = 1, "Buy", f"CCI {val} — oversold"
        elif val > 200:
            score, label, expl = -2, "Strong Sell", f"CCI {val} — extreme overbought"
        elif val > 100:
            score, label, expl = -1, "Sell", f"CCI {val} — overbought"
        elif val > 50:
            score, label, expl = -0.5, "Slightly Bearish", f"CCI {val} — leaning overbought"
        elif val < -50:
            score, label, expl = 0.5, "Slightly Bullish", f"CCI {val} — leaning oversold"
        else:
            score, label, expl = 0, "Neutral", f"CCI {val} — neutral zone"

    cci_name = "CCI (10)" if is_intraday else "CCI (20)"
    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": cci_name, "value": val, "score": score, "signal": label, "explanation": expl, "bet": bet}


def _vwap_signal(df: pd.DataFrame) -> Optional[dict]:
    """VWAP signal — only meaningful for intraday data (multiple rows per date)."""
    # Auto-detect intraday: if multiple rows share the same date, it's intraday
    dates = df.index.date
    if len(set(dates)) == len(dates):
        # Each row is a unique date — daily data, VWAP not meaningful
        return None

    if "VWAP" not in df.columns:
        return None  # No volume — don't add a guaranteed neutral vote

    vwap_val = df["VWAP"].iloc[-1]
    if not _safe(vwap_val):
        return None

    vwap_val = float(vwap_val)
    close = float(df["Close"].iloc[-1])
    diff_pct = (close - vwap_val) / vwap_val * 100 if vwap_val != 0 else 0

    if diff_pct > 0.5:
        score, label = 1, "Buy"
        expl = f"Price {close:.5f} above VWAP {vwap_val:.5f} (+{diff_pct:.2f}%) — bullish intraday"
    elif diff_pct > 0.1:
        score, label = 0.5, "Slightly Bullish"
        expl = f"Price slightly above VWAP {vwap_val:.5f} (+{diff_pct:.2f}%)"
    elif diff_pct < -0.5:
        score, label = -1, "Sell"
        expl = f"Price {close:.5f} below VWAP {vwap_val:.5f} ({diff_pct:.2f}%) — bearish intraday"
    elif diff_pct < -0.1:
        score, label = -0.5, "Slightly Bearish"
        expl = f"Price slightly below VWAP {vwap_val:.5f} ({diff_pct:.2f}%)"
    else:
        score, label = 0, "Neutral"
        expl = f"Price at VWAP {vwap_val:.5f} — balanced"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {"name": "VWAP", "value": round(vwap_val, 5), "score": score, "signal": label, "explanation": expl, "bet": bet}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

INDICATOR_WEIGHTS = {
    "MACD (12,26,9)": 2.0,
    "RSI (14)": 1.5,
    "SMA (20/50/200)": 1.5,
    "ADX (14)": 1.5,
    "EMA (12/26)": 1.0,
    "Bollinger Bands": 1.0,
    "Stochastic (14,3)": 1.0,
    "MFI (14)": 1.5,
    "OBV": 1.0,
    "RSI Divergence": 2.0,
    "MACD Divergence": 2.0,
    "Parabolic SAR": 1.0,
    "CCI (20)": 0.75,
    "VWAP": 1.5,
}

# Weights tuned for intraday — emphasise fast oscillators & VWAP,
# de-emphasise slow trend followers that lose meaning on short bars.
INTRADAY_WEIGHTS = {
    "MACD (8,17,9)": 2.0,
    "RSI (14)": 1.5,
    "SMA (20/50)": 1.0,
    "ADX (10)": 1.0,
    "EMA (12/26)": 1.5,
    "Bollinger Bands": 1.5,
    "Stochastic (5,3)": 1.5,
    "MFI (14)": 1.5,
    "OBV": 1.0,
    "Parabolic SAR": 0.5,
    "CCI (10)": 1.0,
    "VWAP": 2.0,
}

# Trend-following vs mean-reversion classification for regime adjustment
_TREND_FOLLOWING = {
    "MACD (12,26,9)", "MACD (8,17,9)", "EMA (12/26)", "Parabolic SAR",
    "ADX (14)", "ADX (10)", "SMA (20/50/200)", "SMA (20/50)",
}
_MEAN_REVERSION = {
    "RSI (14)", "Bollinger Bands", "Stochastic (14,3)", "Stochastic (5,3)",
    "CCI (20)", "CCI (10)", "MFI (14)", "VWAP",
}


# --- Change 5: Whipsaw detection ---
def _detect_whipsaw(df: pd.DataFrame, is_intraday: bool = False) -> dict:
    """Count MACD histogram sign changes in last 10 bars."""
    if "MACD_Hist" not in df.columns:
        return {"whipsaw_flips": 0, "is_whipsaw": False}
    hist = df["MACD_Hist"].dropna().tail(10)
    if len(hist) < 3:
        return {"whipsaw_flips": 0, "is_whipsaw": False}
    vals = hist.values
    flips = 0
    for i in range(1, len(vals)):
        if _safe(vals[i]) and _safe(vals[i - 1]):
            if (vals[i] > 0 and vals[i - 1] <= 0) or (vals[i] < 0 and vals[i - 1] >= 0):
                flips += 1
    # Higher threshold for intraday — MACD histogram on short bars naturally
    # crosses zero more often; don't penalise normal intraday oscillation
    threshold = 5 if is_intraday else 3
    return {"whipsaw_flips": flips, "is_whipsaw": flips >= threshold}


# --- Change 4: Bidirectional ATR filter ---
def _compute_atr_context(df: pd.DataFrame) -> dict:
    """Compare current ATR to its 50-period SMA to gauge volatility regime."""
    if "ATR" not in df.columns:
        return {"ratio": 1.0, "label": "Normal", "atr_pct": 0}
    atr = df["ATR"].dropna()
    if len(atr) < 50:
        close_val = float(df["Close"].iloc[-1]) if len(df) > 0 else 1
        atr_pct = round(float(atr.iloc[-1]) / close_val * 100, 3) if len(atr) > 0 and close_val != 0 else 0
        return {"ratio": 1.0, "label": "Normal", "atr_pct": atr_pct}

    current_atr = float(atr.iloc[-1])
    atr_sma = float(atr.tail(50).mean())
    close_val = float(df["Close"].iloc[-1])
    ratio = current_atr / atr_sma if atr_sma > 0 else 1.0
    atr_pct = round(current_atr / close_val * 100, 3) if close_val != 0 else 0

    if ratio < 0.6:
        label = "Very Low"
    elif ratio < 0.85:
        label = "Low"
    elif ratio > 2.0:
        label = "Very High"
    elif ratio > 1.5:
        label = "High"
    else:
        label = "Normal"

    return {"ratio": round(ratio, 3), "label": label, "atr_pct": atr_pct}


# --- Change 8: Support/resistance proximity ---
def _compute_sr_context(df: pd.DataFrame) -> dict:
    """Identify support/resistance levels using 5-bar pivots and check proximity."""
    close = float(df["Close"].iloc[-1])
    if "ATR" not in df.columns:
        return {"sr_context": None, "sr_warning": None}
    atr_val = df["ATR"].iloc[-1]
    if not _safe(atr_val):
        return {"sr_context": None, "sr_warning": None}
    atr_val = float(atr_val)

    lookback = min(100, len(df))
    if lookback < 5:
        return {"sr_context": None, "sr_warning": None}

    highs = df["High"].tail(lookback).values
    lows = df["Low"].tail(lookback).values

    resistance_levels = []
    support_levels = []
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            resistance_levels.append(float(highs[i]))
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            support_levels.append(float(lows[i]))

    if not resistance_levels and not support_levels:
        return {"sr_context": None, "sr_warning": None}

    near_threshold = 1.5 * atr_val
    near_resistance = any(abs(close - r) < near_threshold for r in resistance_levels)
    near_support = any(abs(close - s) < near_threshold for s in support_levels)

    sr_context = None
    sr_warning = None
    if near_resistance and near_support:
        sr_context = "Congestion"
        sr_warning = "Price near both support and resistance — congestion zone"
    elif near_resistance:
        sr_context = "Near Resistance"
        sr_warning = "Price approaching resistance level"
    elif near_support:
        sr_context = "Near Support"
        sr_warning = "Price approaching support level"

    return {"sr_context": sr_context, "sr_warning": sr_warning}


# --- Change 9: Signal quality metric ---
def _compute_signal_quality(agreement_pct: float, volatility_label: str,
                            is_whipsaw: bool, score_magnitude: float) -> dict:
    """Composite 0-100 signal quality score."""
    # Agreement contributes 40%
    agreement_component = agreement_pct * 0.4
    # Volatility regime contributes 20%
    vol_scores = {"Normal": 100, "High": 70, "Low": 60, "Very Low": 40, "Very High": 30}
    vol_component = vol_scores.get(volatility_label, 50) * 0.2
    # Whipsaw contributes 20%
    whipsaw_component = (0 if is_whipsaw else 100) * 0.2
    # Score magnitude contributes 20% (higher = more decisive)
    magnitude_component = min(100, score_magnitude / 2 * 100) * 0.2

    quality_score = round(agreement_component + vol_component + whipsaw_component + magnitude_component, 1)
    quality_score = max(0, min(100, quality_score))

    if quality_score >= 75:
        quality_label = "High"
    elif quality_score >= 50:
        quality_label = "Moderate"
    elif quality_score >= 30:
        quality_label = "Low"
    else:
        quality_label = "Very Low"

    return {"score": quality_score, "label": quality_label}


# --- Change 10: Multi-timeframe confirmation ---
def _compute_mtf_context(pair: str, daily_direction: str, interval: str = "1d") -> dict:
    """Fetch higher-timeframe data and compare with current direction.

    For intraday intervals: uses daily data as higher TF.
    For daily: uses weekly data as higher TF (original behavior).
    """
    is_intraday = interval not in ("1d", "1wk", "1mo")

    if is_intraday:
        htf_period, htf_interval, htf_label = "6mo", "1d", "Daily"
        sma_window = 20
        slope_bars = 5
    else:
        htf_period, htf_interval, htf_label = "1y", "1wk", "Weekly"
        sma_window = 20
        slope_bars = 5

    try:
        htf_df = fetch_forex_data(pair, period=htf_period, interval=htf_interval)
    except Exception:
        return {"weekly_trend": "Unknown", "htf_label": htf_label, "warning": None}
    if htf_df is None or len(htf_df) < sma_window:
        return {"weekly_trend": "Unknown", "htf_label": htf_label, "warning": None}

    close = htf_df["Close"]
    sma = close.rolling(window=sma_window).mean()
    if sma.dropna().empty:
        return {"weekly_trend": "Unknown", "htf_label": htf_label, "warning": None}

    current_price = float(close.iloc[-1])
    current_sma = float(sma.iloc[-1])

    sma_recent = sma.dropna().tail(slope_bars)
    sma_slope = float(sma_recent.iloc[-1] - sma_recent.iloc[0]) if len(sma_recent) >= 2 else 0

    if current_price > current_sma and sma_slope > 0:
        htf_trend = "Up"
    elif current_price < current_sma and sma_slope < 0:
        htf_trend = "Down"
    else:
        htf_trend = "Sideways"

    warning = None
    signal_label = "Intraday" if is_intraday else "Daily"
    if daily_direction == "UP" and htf_trend == "Down":
        warning = f"{signal_label} BUY signal conflicts with {htf_label.lower()} downtrend"
    elif daily_direction == "DOWN" and htf_trend == "Up":
        warning = f"{signal_label} SELL signal conflicts with {htf_label.lower()} uptrend"

    return {"weekly_trend": htf_trend, "htf_label": htf_label, "warning": warning}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_indicator_signals(df: pd.DataFrame, interval: str = "1d") -> list[dict]:
    """Compute all individual indicator signals."""
    _intra = interval not in ("1d", "1wk", "1mo")

    signals = []

    def _add(result):
        if result is not None:
            signals.append(result)

    _add(_rsi_signal(df, _intra))
    _add(_macd_signal(df, _intra))
    _add(_sma_signal(df, _intra))
    _add(_ema_signal(df))
    _add(_bollinger_signal(df))
    _add(_stochastic_signal(df, _intra))
    _add(_adx_signal(df, _intra))
    _add(_mfi_signal(df))
    _add(_obv_signal(df))
    # Skip divergence detectors for intraday — 30-bar peak/trough detection
    # is too noisy on sub-daily bars and produces misleading high-weight signals
    if not _intra:
        _add(_rsi_divergence_signal(df))
        _add(_macd_divergence_signal(df))
    _add(_psar_signal(df))
    _add(_cci_signal(df, _intra))
    _add(_vwap_signal(df))

    return signals


def generate_overall_signal(signals: list[dict], df: pd.DataFrame = None,
                            is_intraday: bool = False) -> dict:
    """Weighted average of indicator scores with ATR filter, trend regime,
    confluence, whipsaw, S/R context, and signal quality.

    When ``is_intraday`` is True the dampening factors are gentler and
    the NEUTRAL zone is narrower, preventing the cascading dampening
    from crushing every intraday signal into "mixed".
    """
    if not signals:
        return {
            "score": 0, "signal": "Neutral", "label": "No Data", "strength": 0,
            "direction": "NEUTRAL", "up_count": 0, "down_count": 0,
            "neutral_count": 0, "agreement_pct": 0, "summary": "No data available",
            "confluence_level": "None", "trend_regime": "Unknown", "volatility": "Unknown",
            "atr_pct": 0, "whipsaw_flips": 0, "is_whipsaw": False,
            "sr_context": None, "sr_warning": None,
            "signal_quality": {"score": 0, "label": "Very Low"},
        }

    weights = INTRADAY_WEIGHTS if is_intraday else INDICATOR_WEIGHTS

    # --- Count directions ---
    up_count = 0
    down_count = 0
    neutral_count = 0
    for s in signals:
        if s["score"] > 0:
            up_count += 1
        elif s["score"] < 0:
            down_count += 1
        else:
            neutral_count += 1

    # --- B. Trend regime adjustment (ADX-based) ---
    adx_val = None
    if df is not None and "ADX" in df.columns:
        raw_adx = df["ADX"].iloc[-1]
        if _safe(raw_adx):
            adx_val = float(raw_adx)
    trend_threshold = 20 if is_intraday else 25
    trending = adx_val is not None and adx_val > trend_threshold
    trend_regime = "Trending" if trending else "Ranging"

    # --- Weighted sum with regime adjustment ---
    weighted_sum = 0.0
    total_weight = 0.0
    for s in signals:
        w = weights.get(s["name"], 1.0)
        # Apply trend regime multiplier
        if adx_val is not None:
            if s["name"] in _TREND_FOLLOWING:
                w *= 1.2 if trending else 0.85
            elif s["name"] in _MEAN_REVERSION:
                w *= 0.85 if trending else 1.2
        weighted_sum += s["score"] * w
        total_weight += w

    avg = weighted_sum / total_weight if total_weight else 0

    # --- A. ATR volatility filter (bidirectional) ---
    # Gentler multipliers for intraday — intraday ATR variance is normal
    atr_ctx = _compute_atr_context(df) if df is not None else {"ratio": 1.0, "label": "Normal", "atr_pct": 0}
    if is_intraday:
        if atr_ctx["ratio"] < 0.6:
            avg *= 0.8
        elif atr_ctx["ratio"] > 2.0:
            avg *= 0.8
    else:
        if atr_ctx["ratio"] < 0.6:
            avg *= 0.5
        elif atr_ctx["ratio"] < 0.85:
            avg *= 0.75
        elif atr_ctx["ratio"] > 2.0:
            avg *= 0.5
        elif atr_ctx["ratio"] > 1.5:
            avg *= 0.75

    # --- Whipsaw detection ---
    whipsaw = _detect_whipsaw(df, is_intraday) if df is not None else {"whipsaw_flips": 0, "is_whipsaw": False}
    if whipsaw["is_whipsaw"]:
        avg *= 0.8 if is_intraday else 0.6

    # --- C. Confluence (5-tier) ---
    total = len(signals)
    directional = total - neutral_count
    if directional > 0:
        majority = max(up_count, down_count)
        agreement_ratio = majority / directional
    else:
        agreement_ratio = 0

    agreement_pct_raw = agreement_ratio * 100

    if whipsaw["is_whipsaw"]:
        # Override confluence during whipsaw
        confluence_level = "Low"
    elif agreement_pct_raw >= 85:
        avg *= 1.25
        confluence_level = "Very High"
    elif agreement_pct_raw >= 70:
        avg *= 1.15
        confluence_level = "High"
    elif agreement_pct_raw >= 50:
        confluence_level = "Moderate"
    elif agreement_pct_raw >= 35:
        avg *= 0.90 if is_intraday else 0.80
        confluence_level = "Low"
    else:
        avg *= 0.80 if is_intraday else 0.60
        confluence_level = "Very Low"

    # --- S/R context adjustment ---
    sr_data = _compute_sr_context(df) if df is not None else {"sr_context": None, "sr_warning": None}

    # Determine preliminary direction for S/R adjustment
    neutral_threshold = 0.15 if is_intraday else 0.25
    if avg > neutral_threshold:
        prelim_direction = "UP"
    elif avg < -neutral_threshold:
        prelim_direction = "DOWN"
    else:
        prelim_direction = "NEUTRAL"

    if sr_data["sr_context"] is not None and prelim_direction != "NEUTRAL":
        if prelim_direction == "UP" and sr_data["sr_context"] == "Near Resistance":
            avg *= 0.85
            sr_data["sr_warning"] = "Buy signal near resistance — dampened"
        elif prelim_direction == "DOWN" and sr_data["sr_context"] == "Near Support":
            avg *= 0.85
            sr_data["sr_warning"] = "Sell signal near support — dampened"
        elif prelim_direction == "UP" and sr_data["sr_context"] == "Near Support":
            avg *= 1.1
            sr_data["sr_warning"] = "Buy signal at support — reinforced"
        elif prelim_direction == "DOWN" and sr_data["sr_context"] == "Near Resistance":
            avg *= 1.1
            sr_data["sr_warning"] = "Sell signal at resistance — reinforced"

    # --- Label --- (lower thresholds for intraday)
    if is_intraday:
        if avg >= 1.0:
            label = "Strong Buy"
        elif avg >= 0.3:
            label = "Buy"
        elif avg <= -1.0:
            label = "Strong Sell"
        elif avg <= -0.3:
            label = "Sell"
        else:
            label = "Neutral"
    else:
        if avg >= 1.5:
            label = "Strong Buy"
        elif avg >= 0.5:
            label = "Buy"
        elif avg <= -1.5:
            label = "Strong Sell"
        elif avg <= -0.5:
            label = "Sell"
        else:
            label = "Neutral"

    # Verdict direction
    if avg > neutral_threshold:
        direction = "UP"
    elif avg < -neutral_threshold:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    agreement_pct = round(max(up_count, down_count) / total * 100, 1) if total else 0
    majority = max(up_count, down_count)
    majority_dir = "UP" if up_count >= down_count else "DOWN"
    if direction == "NEUTRAL":
        summary = "Indicators are mixed — no clear direction"
    else:
        summary = f"{majority} of {total} indicators suggest price may go {majority_dir}"

    # --- Signal quality ---
    signal_quality = _compute_signal_quality(
        agreement_pct=agreement_pct,
        volatility_label=atr_ctx["label"],
        is_whipsaw=whipsaw["is_whipsaw"],
        score_magnitude=abs(avg),
    )

    return {
        "score": round(avg, 3), "signal": label,
        "strength": round(abs(avg) / 2 * 100, 1),
        "direction": direction, "up_count": up_count, "down_count": down_count,
        "neutral_count": neutral_count, "agreement_pct": agreement_pct,
        "summary": summary,
        "confluence_level": confluence_level,
        "trend_regime": trend_regime,
        "volatility": atr_ctx["label"],
        "atr_pct": atr_ctx["atr_pct"],
        "whipsaw_flips": whipsaw["whipsaw_flips"],
        "is_whipsaw": whipsaw["is_whipsaw"],
        "sr_context": sr_data["sr_context"],
        "sr_warning": sr_data["sr_warning"],
        "signal_quality": signal_quality,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_pair(pair: str, period: str = "6mo", interval: str = "1d") -> Optional[dict]:
    """Full analysis for a currency pair — returns everything the frontend needs."""
    df = fetch_forex_data(pair, period, interval)
    if df is None or len(df) < 20:
        return None

    _intra = interval not in ("1d", "1wk", "1mo")
    df = compute_indicators(df, interval)

    signals = get_indicator_signals(df, interval)
    overall = generate_overall_signal(signals, df, is_intraday=_intra)

    # --- MTF confirmation ---
    mtf = _compute_mtf_context(pair, overall["direction"], interval=interval)

    # Apply MTF dampening if conflict detected (gentler for intraday)
    if mtf["warning"] is not None:
        mtf_factor = 0.85 if _intra else 0.7
        overall["score"] = round(overall["score"] * mtf_factor, 3)
        overall["strength"] = round(abs(overall["score"]) / 2 * 100, 1)
        # Re-derive label after dampening
        s = overall["score"]
        if _intra:
            if s >= 1.0:
                overall["signal"] = "Strong Buy"
            elif s >= 0.3:
                overall["signal"] = "Buy"
            elif s <= -1.0:
                overall["signal"] = "Strong Sell"
            elif s <= -0.3:
                overall["signal"] = "Sell"
            else:
                overall["signal"] = "Neutral"
        else:
            if s >= 1.5:
                overall["signal"] = "Strong Buy"
            elif s >= 0.5:
                overall["signal"] = "Buy"
            elif s <= -1.5:
                overall["signal"] = "Strong Sell"
            elif s <= -0.5:
                overall["signal"] = "Sell"
            else:
                overall["signal"] = "Neutral"
        # Re-derive direction
        neutral_threshold = 0.15 if _intra else 0.25
        if s > neutral_threshold:
            overall["direction"] = "UP"
        elif s < -neutral_threshold:
            overall["direction"] = "DOWN"
        else:
            overall["direction"] = "NEUTRAL"

    # Prepare chart data (OHLC)
    chart_data = []
    for ts, row in df.iterrows():
        t = int(ts.timestamp())
        chart_data.append({
            "time": t,
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5),
        })

    # Overlay line data
    def _series(col: str) -> list[dict]:
        out = []
        for ts, val in df[col].dropna().items():
            out.append({"time": int(ts.timestamp()), "value": round(float(val), 5)})
        return out

    overlays = {
        "sma20": _series("SMA_20"),
        "sma50": _series("SMA_50"),
        "sma200": _series("SMA_200"),
        "ema12": _series("EMA_12"),
        "ema26": _series("EMA_26"),
        "ema9": _series("EMA_9"),
        "ema21": _series("EMA_21"),
        "bb_upper": _series("BB_Upper"),
        "bb_middle": _series("BB_Middle"),
        "bb_lower": _series("BB_Lower"),
        "vwap": _series("VWAP") if "VWAP" in df.columns else [],
    }

    # Sub-chart data
    rsi_data = _series("RSI")
    macd_data = _series("MACD")
    macd_signal_data = _series("MACD_Signal")
    macd_hist_data = _series("MACD_Hist")
    stoch_k_data = _series("Stoch_K")
    stoch_d_data = _series("Stoch_D")

    last_close = round(float(df["Close"].iloc[-1]), 5)
    prev_close = round(float(df["Close"].iloc[-2]), 5) if len(df) > 1 else last_close
    change = round(last_close - prev_close, 5)
    change_pct = round(change / prev_close * 100, 3) if prev_close else 0

    return {
        "pair": pair,
        "period": period,
        "interval": interval,
        "last_price": last_close,
        "change": change,
        "change_pct": change_pct,
        "overall": overall,
        "signals": signals,
        "chart": chart_data,
        "overlays": overlays,
        "rsi": rsi_data,
        "macd": macd_data,
        "macd_signal": macd_signal_data,
        "macd_hist": macd_hist_data,
        "stoch_k": stoch_k_data,
        "stoch_d": stoch_d_data,
        "mtf": mtf,
    }
