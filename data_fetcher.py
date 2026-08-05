import math
import os
import time
from threading import Lock
from typing import Optional

import pandas as pd
import requests

TWELVE_DATA_API_KEY = os.environ.get(
    "TWELVE_DATA_API_KEY", "f13f6001aa72470c8578c87db582328c"
)
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

CURRENCY_PAIRS = {
    "INR": {
        "USD/INR": "USD/INR",
        "EUR/INR": "EUR/INR",
        "GBP/INR": "GBP/INR",
        "JPY/INR": "JPY/INR",
    },
    "Major": {
        "EUR/USD": "EUR/USD",
        "GBP/USD": "GBP/USD",
        "USD/JPY": "USD/JPY",
        "AUD/USD": "AUD/USD",
        "USD/CHF": "USD/CHF",
        "NZD/USD": "NZD/USD",
        "USD/CAD": "USD/CAD",
    },
    "Commodities": {
        "XAU/USD": "XAU/USD",
    },
}

# Flatten for quick lookup: "USD/INR" -> "USD/INR"
ALL_PAIRS: dict[str, str] = {}
for group in CURRENCY_PAIRS.values():
    ALL_PAIRS.update(group)

# Twelve Data interval names
_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "1d": "1day",
    "1wk": "1week",
}

# Period string -> approximate calendar days
_PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "60d": 60,
}

# Interval -> minutes per bar
_INTERVAL_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "1d": 1440,
    "1wk": 10080,
}

# Interval-specific cache TTLs (seconds)
_CACHE_TTLS = {
    "1m": 15,
    "5m": 30,
    "15m": 60,
    "30m": 90,
    "1h": 120,
    "1d": 300,
    "1wk": 600,
}

# In-memory cache: key -> (timestamp, DataFrame)
_cache: dict[str, tuple[float, pd.DataFrame]] = {}

# Global rate limiting
_last_api_call = 0.0
_api_call_lock = Lock()
MIN_API_CALL_INTERVAL = 8.0  # 8 seconds between calls (free tier: 8 requests/min)


def _compute_outputsize(period: str, interval: str) -> int:
    """Convert a yfinance-style period + interval into Twelve Data outputsize (bar count)."""
    days = _PERIOD_TO_DAYS.get(period, 180)
    interval_minutes = _INTERVAL_TO_MINUTES.get(interval, 1440)
    trading_days = days * 5 / 7
    bars = int(math.ceil(trading_days * 24 * 60 / interval_minutes))
    return max(30, min(5000, bars))


def fetch_forex_data(
    pair: str,
    period: str = "6mo",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Fetch OHLC data for a currency pair via Twelve Data REST API with caching."""
    symbol = ALL_PAIRS.get(pair)
    if symbol is None:
        return None

    td_interval = _INTERVAL_MAP.get(interval, "1day")
    outputsize = _compute_outputsize(period, interval)
    cache_key = f"{symbol}_{period}_{interval}"
    now = time.time()
    ttl = _CACHE_TTLS.get(interval, 300)

    if cache_key in _cache:
        cached_time, cached_df = _cache[cache_key]
        if now - cached_time < ttl:
            return cached_df.copy()

    try:
        # Rate limiting: ensure minimum interval between API calls
        global _last_api_call
        with _api_call_lock:
            time_since_last_call = now - _last_api_call
            if time_since_last_call < MIN_API_CALL_INTERVAL:
                sleep_time = MIN_API_CALL_INTERVAL - time_since_last_call
                print(f"Rate limiting: waiting {sleep_time:.1f}s before API call for {pair}")
                time.sleep(sleep_time)

            _last_api_call = time.time()
            resp = requests.get(
                TWELVE_DATA_BASE_URL,
                params={
                    "symbol": symbol,
                    "interval": td_interval,
                    "outputsize": outputsize,
                    "apikey": TWELVE_DATA_API_KEY,
                },
                timeout=10,
            )

        # Rate limit — serve stale cache if available
        if resp.status_code == 429:
            print(f"Rate limited fetching {pair}")
            if cache_key in _cache:
                cached_time, cached_df = _cache[cache_key]
                age_hours = (now - cached_time) / 3600
                print(f"Serving stale cache for {pair} (age: {age_hours:.1f}h)")
                return cached_df.copy()
            print(f"No stale cache available for {pair}")
            return None

        resp.raise_for_status()
        data = resp.json()

        if "values" not in data:
            error_msg = data.get("message", "Unknown error")
            print(f"Twelve Data error for {pair}: {error_msg}")
            if cache_key in _cache:
                return _cache[cache_key][1].copy()
            return None

        df = pd.DataFrame(data["values"])
        if df.empty:
            return None

        # Rename columns to match expected OHLC format
        df.rename(
            columns={
                "datetime": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        # Convert to numeric
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        # Set DatetimeIndex, sort ascending
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        df.sort_index(inplace=True)

        # Keep OHLC + Volume
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
        print(f"Error fetching {pair} ({symbol}): {e}")
        # Serve stale cache on network errors
        if cache_key in _cache:
            return _cache[cache_key][1].copy()
        return None
