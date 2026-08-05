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

    # MACD — faster (8,17,9) for intraday; standard (12,26,9) for daily
    try:
        mf, ms, msig = (8, 17, 9) if _intra else (12, 26, 9)
        macd = ta.trend.MACD(close, window_slow=ms, window_fast=mf, window_sign=msig)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
    except Exception:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = float("nan")

    # EMAs
    _try_indicator(df, "EMA_12", lambda: ta.trend.EMAIndicator(close, window=12).ema_indicator())
    _try_indicator(df, "EMA_26", lambda: ta.trend.EMAIndicator(close, window=26).ema_indicator())
    _try_indicator(df, "EMA_9", lambda: ta.trend.EMAIndicator(close, window=9).ema_indicator())
    _try_indicator(df, "EMA_21", lambda: ta.trend.EMAIndicator(close, window=21).ema_indicator())

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
        # VWAP — Volume Weighted Average Price (daily reset)
        try:
            typical_price = (high + low + close) / 3
            dates = df.index.date
            cum_tp_vol = (typical_price * volume).groupby(dates).cumsum()
            cum_vol = volume.groupby(dates).cumsum()
            df["VWAP"] = cum_tp_vol / cum_vol.replace(0, float("nan"))
        except Exception:
            df["VWAP"] = float("nan")

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
    "EMA (12/26)": 1.0,
    "ADX (14)": 1.5,
    "VWAP": 1.5,
    "SMC": 2.0,
}

# Weights tuned for intraday — emphasise fast oscillators & VWAP,
# de-emphasise slow trend followers that lose meaning on short bars.
INTRADAY_WEIGHTS = {
    "MACD (8,17,9)": 2.0,
    "EMA (12/26)": 1.5,
    "ADX (10)": 1.0,
    "VWAP": 2.0,
    "SMC": 2.5,
}

# Trend-following vs mean-reversion classification for regime adjustment
_TREND_FOLLOWING = {
    "MACD (12,26,9)", "MACD (8,17,9)", "EMA (12/26)",
    "ADX (14)", "ADX (10)", "SMC",
}
_MEAN_REVERSION = {
    "VWAP",
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


# ---------------------------------------------------------------------------
# SMC / ICT structure detection
# ---------------------------------------------------------------------------

def _detect_swings(df: pd.DataFrame, pivot_n: int = 5) -> dict:
    """Find swing highs and swing lows using N-bar pivot logic."""
    highs = df["High"].values
    lows = df["Low"].values
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(pivot_n, len(df) - pivot_n):
        if all(highs[i] >= highs[i - j] for j in range(1, pivot_n + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, pivot_n + 1)):
            swing_highs.append((i, float(highs[i])))
        if all(lows[i] <= lows[i - j] for j in range(1, pivot_n + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, pivot_n + 1)):
            swing_lows.append((i, float(lows[i])))
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def compute_smc(df: pd.DataFrame, interval: str = "1d") -> dict:
    """Compute all SMC/ICT structures: FVG, BOS, CHoCH, liquidity sweeps, order blocks."""
    _intra = interval not in ("1d", "1wk", "1mo")
    pivot_n = 3 if _intra else 5
    swings = _detect_swings(df, pivot_n)
    n_bars = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values

    # --- Fair Value Gaps (FVG) ---
    fvg_list: list[dict] = []
    for i in range(1, n_bars - 1):
        # Bullish FVG: candle[i+1].low > candle[i-1].high  (gap up)
        if lows[i + 1] > highs[i - 1]:
            fvg_list.append({
                "type": "bullish",
                "top": float(lows[i + 1]),
                "bottom": float(highs[i - 1]),
                "start_idx": i - 1,
                "end_idx": i + 1,
                "filled": False,
            })
        # Bearish FVG: candle[i+1].high < candle[i-1].low  (gap down)
        elif highs[i + 1] < lows[i - 1]:
            fvg_list.append({
                "type": "bearish",
                "top": float(lows[i - 1]),
                "bottom": float(highs[i + 1]),
                "start_idx": i - 1,
                "end_idx": i + 1,
                "filled": False,
            })

    # Check if FVGs have been filled by subsequent price action
    for fvg in fvg_list:
        for j in range(fvg["end_idx"] + 1, n_bars):
            if fvg["type"] == "bullish" and lows[j] <= fvg["bottom"]:
                fvg["filled"] = True
                break
            elif fvg["type"] == "bearish" and highs[j] >= fvg["top"]:
                fvg["filled"] = True
                break

    fvg_list = fvg_list[-20:]  # keep last 20

    # --- BOS / CHoCH detection ---
    bos_choch: list[dict] = []
    sh = swings["swing_highs"]
    sl = swings["swing_lows"]

    # Determine trend from consecutive swing highs/lows
    trend = "neutral"  # "up", "down", "neutral"
    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1][1] > sh[-2][1]  # higher high
        hl = sl[-1][1] > sl[-2][1]  # higher low
        ll = sl[-1][1] < sl[-2][1]  # lower low
        lh = sh[-1][1] < sh[-2][1]  # lower high
        if hh and hl:
            trend = "up"
        elif ll and lh:
            trend = "down"

    # Scan for breaks of structure
    for i in range(n_bars):
        # Check against recent swing highs
        for sh_idx, sh_price in sh:
            if sh_idx >= i:
                continue
            if i - sh_idx > 50:
                continue
            if highs[i] > sh_price and closes[i] > sh_price:
                if trend == "up":
                    bos_choch.append({
                        "type": "BOS", "direction": "bullish",
                        "level": sh_price, "bar_index": i,
                    })
                elif trend == "down":
                    bos_choch.append({
                        "type": "CHoCH", "direction": "bullish",
                        "level": sh_price, "bar_index": i,
                    })
                break  # one event per bar

        # Check against recent swing lows
        for sl_idx, sl_price in sl:
            if sl_idx >= i:
                continue
            if i - sl_idx > 50:
                continue
            if lows[i] < sl_price and closes[i] < sl_price:
                if trend == "down":
                    bos_choch.append({
                        "type": "BOS", "direction": "bearish",
                        "level": sl_price, "bar_index": i,
                    })
                elif trend == "up":
                    bos_choch.append({
                        "type": "CHoCH", "direction": "bearish",
                        "level": sl_price, "bar_index": i,
                    })
                break

    # Deduplicate: keep last event per bar_index
    seen_bars: set[int] = set()
    deduped: list[dict] = []
    for evt in reversed(bos_choch):
        if evt["bar_index"] not in seen_bars:
            seen_bars.add(evt["bar_index"])
            deduped.append(evt)
    bos_choch = list(reversed(deduped))[-5:]

    # --- Liquidity Sweeps ---
    liquidity_sweeps: list[dict] = []
    for i in range(1, n_bars):
        # Check against swing lows from last 50 bars
        for sl_idx, sl_price in sl:
            if i - sl_idx > 50 or sl_idx >= i:
                continue
            # Bullish sweep: low goes below swing low but close is above it
            if lows[i] < sl_price and closes[i] > sl_price:
                liquidity_sweeps.append({
                    "type": "bullish", "swept_level": sl_price, "bar_index": i,
                })
                break
        # Check against swing highs from last 50 bars
        for sh_idx, sh_price in sh:
            if i - sh_idx > 50 or sh_idx >= i:
                continue
            # Bearish sweep: high goes above swing high but close is below it
            if highs[i] > sh_price and closes[i] < sh_price:
                liquidity_sweeps.append({
                    "type": "bearish", "swept_level": sh_price, "bar_index": i,
                })
                break

    liquidity_sweeps = liquidity_sweeps[-10:]

    # --- Order Blocks ---
    order_blocks: list[dict] = []
    for evt in bos_choch:
        if evt["type"] != "BOS":
            continue
        bi = evt["bar_index"]
        if evt["direction"] == "bullish":
            # Bullish OB: last bearish candle before the bullish BOS
            for k in range(bi - 1, max(bi - 10, -1), -1):
                if k < 0:
                    break
                if closes[k] < opens[k]:  # bearish candle
                    order_blocks.append({
                        "type": "bullish",
                        "top": float(highs[k]),
                        "bottom": float(lows[k]),
                        "bar_index": k,
                        "tested": False,
                    })
                    break
        elif evt["direction"] == "bearish":
            # Bearish OB: last bullish candle before the bearish BOS
            for k in range(bi - 1, max(bi - 10, -1), -1):
                if k < 0:
                    break
                if closes[k] > opens[k]:  # bullish candle
                    order_blocks.append({
                        "type": "bearish",
                        "top": float(highs[k]),
                        "bottom": float(lows[k]),
                        "bar_index": k,
                        "tested": False,
                    })
                    break

    # Check if OBs have been tested (price returned to zone)
    for ob in order_blocks:
        for j in range(ob["bar_index"] + 1, n_bars):
            if lows[j] <= ob["top"] and highs[j] >= ob["bottom"]:
                ob["tested"] = True
                break

    order_blocks = order_blocks[-5:]

    # --- Bias ---
    bias = "neutral"
    if bos_choch:
        last_evt = bos_choch[-1]
        if last_evt["direction"] == "bullish":
            bias = "bullish"
        elif last_evt["direction"] == "bearish":
            bias = "bearish"
    elif trend == "up":
        bias = "bullish"
    elif trend == "down":
        bias = "bearish"

    return {
        "fvg": fvg_list,
        "bos_choch": bos_choch,
        "liquidity_sweeps": liquidity_sweeps,
        "order_blocks": order_blocks,
        "bias": bias,
    }


def _smc_signal(df: pd.DataFrame, smc_data: dict) -> Optional[dict]:
    """Aggregate SMC structures into a single signal dict."""
    n_bars = len(df)
    score = 0.0
    parts: list[str] = []
    closes = df["Close"].values
    last_close = float(closes[-1])

    # BOS / CHoCH (last 5 bars)
    for evt in smc_data["bos_choch"]:
        if n_bars - evt["bar_index"] > 5:
            continue
        if evt["type"] == "BOS":
            if evt["direction"] == "bullish":
                score += 0.75
                parts.append("Bullish BOS")
            else:
                score -= 0.75
                parts.append("Bearish BOS")
        elif evt["type"] == "CHoCH":
            if evt["direction"] == "bullish":
                score += 1.0
                parts.append("Bullish CHoCH")
            else:
                score -= 1.0
                parts.append("Bearish CHoCH")

    # Unfilled FVGs — price in/near zone
    for fvg in smc_data["fvg"]:
        if fvg["filled"]:
            continue
        # Check if current price is within or near the FVG zone
        zone_height = abs(fvg["top"] - fvg["bottom"])
        near_margin = zone_height * 0.5
        if fvg["type"] == "bullish":
            if fvg["bottom"] - near_margin <= last_close <= fvg["top"] + near_margin:
                score += 0.5
                parts.append("Unfilled bullish FVG")
                break  # only count one
        elif fvg["type"] == "bearish":
            if fvg["bottom"] - near_margin <= last_close <= fvg["top"] + near_margin:
                score -= 0.5
                parts.append("Unfilled bearish FVG")
                break

    # Liquidity sweeps (last 3 bars)
    for ls in smc_data["liquidity_sweeps"]:
        if n_bars - ls["bar_index"] > 3:
            continue
        if ls["type"] == "bullish":
            score += 0.5
            parts.append("Bullish liquidity sweep")
        else:
            score -= 0.5
            parts.append("Bearish liquidity sweep")

    # Order blocks — price at OB zone
    for ob in smc_data["order_blocks"]:
        if ob["bottom"] <= last_close <= ob["top"]:
            if ob["type"] == "bullish":
                score += 0.5
                parts.append("Bullish OB")
            else:
                score -= 0.5
                parts.append("Bearish OB")
            break  # only count one

    # Clamp to [-2, +2]
    score = max(-2.0, min(2.0, score))
    score = round(score, 2)

    if not parts:
        expl = "No active SMC structures"
        label = "Neutral"
    else:
        expl = " + ".join(parts)
        if score >= 1.5:
            label = "Strong Buy"
        elif score > 0:
            label = "Buy" if score >= 0.5 else "Slightly Bullish"
        elif score <= -1.5:
            label = "Strong Sell"
        elif score < 0:
            label = "Sell" if score <= -0.5 else "Slightly Bearish"
        else:
            label = "Neutral"

    bet = "UP" if score > 0 else ("DOWN" if score < 0 else "--")
    return {
        "name": "SMC", "value": smc_data["bias"].title(),
        "score": score, "signal": label, "explanation": expl, "bet": bet,
    }


def _compute_smc_context(smc_data: dict, direction: str) -> float:
    """Return a multiplier for SMC dampening/reinforcement of the overall score."""
    bias = smc_data["bias"]
    if bias == "neutral":
        return 1.0

    # Check for recent CHoCH (strong reversal signal)
    has_recent_choch = any(e["type"] == "CHoCH" for e in smc_data["bos_choch"])

    if direction == "UP" and bias == "bearish" and has_recent_choch:
        return 0.8
    elif direction == "DOWN" and bias == "bullish" and has_recent_choch:
        return 0.8
    elif direction == "UP" and bias == "bullish":
        return 1.15
    elif direction == "DOWN" and bias == "bearish":
        return 1.15

    return 1.0


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

def get_indicator_signals(df: pd.DataFrame, interval: str = "1d",
                          smc_data: Optional[dict] = None) -> list[dict]:
    """Compute all individual indicator signals."""
    _intra = interval not in ("1d", "1wk", "1mo")

    signals = []

    def _add(result):
        if result is not None:
            signals.append(result)

    _add(_macd_signal(df, _intra))
    _add(_ema_signal(df))
    _add(_adx_signal(df, _intra))
    _add(_vwap_signal(df))
    if smc_data is not None:
        _add(_smc_signal(df, smc_data))

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

    # --- SMC / ICT structure detection ---
    smc_data = compute_smc(df, interval)

    signals = get_indicator_signals(df, interval, smc_data=smc_data)
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

    # --- SMC context dampening / reinforcement ---
    smc_factor = _compute_smc_context(smc_data, overall["direction"])
    if smc_factor != 1.0:
        overall["score"] = round(overall["score"] * smc_factor, 3)
        overall["strength"] = round(abs(overall["score"]) / 2 * 100, 1)
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
        "ema12": _series("EMA_12"),
        "ema26": _series("EMA_26"),
        "ema9": _series("EMA_9"),
        "ema21": _series("EMA_21"),
        "vwap": _series("VWAP") if "VWAP" in df.columns else [],
    }

    # Sub-chart data
    macd_data = _series("MACD")
    macd_signal_data = _series("MACD_Signal")
    macd_hist_data = _series("MACD_Hist")

    last_close = round(float(df["Close"].iloc[-1]), 5)
    prev_close = round(float(df["Close"].iloc[-2]), 5) if len(df) > 1 else last_close
    change = round(last_close - prev_close, 5)
    change_pct = round(change / prev_close * 100, 3) if prev_close else 0

    # SMC chart data for frontend rendering
    smc_chart_data = {
        "fvg": [{"type": f["type"], "top": f["top"], "bottom": f["bottom"],
                 "start": int(df.index[f["start_idx"]].timestamp()),
                 "end": int(df.index[min(f["end_idx"], len(df) - 1)].timestamp()),
                 "filled": f["filled"]} for f in smc_data["fvg"][-10:]],
        "bos_choch": [{"type": e["type"], "direction": e["direction"],
                       "level": e["level"],
                       "time": int(df.index[e["bar_index"]].timestamp())} for e in smc_data["bos_choch"][-5:]],
        "order_blocks": [{"type": ob["type"], "top": ob["top"], "bottom": ob["bottom"],
                          "time": int(df.index[ob["bar_index"]].timestamp()),
                          "tested": ob["tested"]} for ob in smc_data["order_blocks"][-5:]],
        "liquidity_sweeps": [{"type": ls["type"],
                              "level": ls["swept_level"],
                              "time": int(df.index[ls["bar_index"]].timestamp())} for ls in smc_data["liquidity_sweeps"][-5:]],
        "bias": smc_data["bias"],
    }

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
        "macd": macd_data,
        "macd_signal": macd_signal_data,
        "macd_hist": macd_hist_data,
        "mtf": mtf,
        "smc": smc_chart_data,
    }
