import time
from typing import Optional

import pandas as pd
import yfinance as yf

CURRENCY_PAIRS = {
    "INR": {
        "USD/INR": "USDINR=X",
        "EUR/INR": "EURINR=X",
        "GBP/INR": "GBPINR=X",
        "JPY/INR": "JPYINR=X",
    },
    "Major": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CHF": "USDCHF=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CAD": "USDCAD=X",
    },
    "Commodities": {
        "XAU/USD": "GC=F",
    },
}

# Flatten for quick lookup: "USD/INR" -> "USDINR=X"
ALL_PAIRS: dict[str, str] = {}
for group in CURRENCY_PAIRS.values():
    ALL_PAIRS.update(group)

# In-memory cache: key -> (timestamp, DataFrame)
_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 300  # 5 minutes


def fetch_forex_data(
    pair: str,
    period: str = "6mo",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Fetch OHLC data for a currency pair via yfinance with caching."""
    ticker = ALL_PAIRS.get(pair)
    if ticker is None:
        return None

    cache_key = f"{ticker}_{period}_{interval}"
    now = time.time()
    ttl = 60 if interval in ("1m", "5m") else CACHE_TTL

    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if now - cached_time < ttl:
            return cached_df.copy()

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Keep OHLC + Volume (if available) columns
        keep = ["Open", "High", "Low", "Close"]
        if "Volume" in df.columns:
            keep.append("Volume")
        df = df[keep].copy()
        df.dropna(inplace=True)
        if df.empty:
            return None
        _cache[cache_key] = (now, df)
        return df.copy()
    except Exception as e:
        print(f"Error fetching {pair} ({ticker}): {e}")
        return None
