#!/usr/bin/env python3
"""Test TwelveData API connection and pre-populate cache."""

import os
import time
from data_fetcher import fetch_forex_data, ALL_PAIRS

def test_api_key():
    """Test if the API key is working."""
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key or api_key == "f13f6001aa72470c8578c87db582328c":
        print("⚠️  WARNING: Using default demo API key")
        print("   Get your free key at: https://twelvedata.com/")
        print("   Add it to .env: TWELVE_DATA_API_KEY=your_key_here")
        return False
    else:
        print(f"✅ Using custom API key: {api_key[:8]}...")
        return True

def warm_up_cache():
    """Pre-populate cache with most common pairs."""
    essential_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/INR",
        "AUD/USD", "USD/CHF", "USD/CAD", "XAU/USD"
    ]

    print("\n🔄 Warming up cache with essential pairs...")

    for pair in essential_pairs:
        if pair in ALL_PAIRS:
            print(f"Fetching {pair}...")
            try:
                df = fetch_forex_data(pair, period="5d", interval="1d")
                if df is not None:
                    print(f"  ✅ {pair}: {len(df)} bars cached")
                else:
                    print(f"  ❌ {pair}: Failed to fetch")
            except Exception as e:
                print(f"  ❌ {pair}: Error - {e}")

            # Small delay to respect rate limits
            time.sleep(1)

if __name__ == "__main__":
    print("TwelveData API Test\n" + "=" * 20)

    has_custom_key = test_api_key()

    if has_custom_key:
        warm_up_cache()
        print("\n✅ Cache warmed up! Your dashboard should work better now.")
    else:
        print("\n⚠️  Please set up your API key first:")
        print("   1. Get free key: https://twelvedata.com/")
        print("   2. Add to .env: TWELVE_DATA_API_KEY=your_key_here")
        print("   3. Run this script again")