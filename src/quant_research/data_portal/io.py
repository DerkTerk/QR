import os
from pathlib import Path
import pandas as pd

from quant_research.data_portal.eikon_loader import fetch_prices_volumes
from quant_research.processing.clean_intraday import clean_to_rth

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _processed_path(ticker, start, end):
    """
    Build path for processed parquet file.
    Example: data/processed/AAPL_2024-08-01_2024-08-31.parquet
    """
    safe_ticker = str(ticker).replace("/", "_").replace(":", "_")
    return PROCESSED_DIR / f"{safe_ticker}_{start}_{end}.parquet"


def _raw_path(ticker, start, end, kind):
    """
    Build path for raw parquet file (prices or volumes).
    Example: data/raw/AAPL_prices_2024-08-01_2024-08-31.parquet
    """
    safe_ticker = str(ticker).replace("/", "_").replace(":", "_")
    return RAW_DIR / f"{safe_ticker}_{kind}_{start}_{end}.parquet"


def load_or_fetch_processed(
    tickers, start, end, interval="minute", force_refresh=False, save_raw=False
):
    prices_dict, volumes_dict = {}, {}

    for ticker in tickers:
        p_path = _processed_path(ticker, start, end)

        if p_path.exists() and not force_refresh:
            df = pd.read_parquet(p_path)
            prices_dict[ticker] = df.get(
                "price", pd.Series(index=df.index, dtype="float64")
            )
            volumes_dict[ticker] = df.get(
                "volume", pd.Series(index=df.index, dtype="float64")
            )
            continue

        # Fetch raw
        raw_prices, raw_volumes = fetch_prices_volumes(
            [ticker], start=start, end=end, interval=interval
        )

        if save_raw:
            (
                raw_prices[[ticker]] if ticker in raw_prices.columns else raw_prices
            ).to_parquet(_raw_path(ticker, start, end, "prices"))
            (
                raw_volumes[[ticker]] if ticker in raw_volumes.columns else raw_volumes
            ).to_parquet(_raw_path(ticker, start, end, "volumes"))

        # Process depending on interval
        intraday = interval.lower() in ("minute", "1m")
        if intraday:
            cleaned_prices, cleaned_volumes = clean_to_rth(raw_prices, raw_volumes)
        else:
            # Daily: just align indices; no RTH filter
            cleaned_prices, cleaned_volumes = (
                raw_prices.sort_index(),
                raw_volumes.sort_index(),
            )
            idx = cleaned_prices.index.union(cleaned_volumes.index)
            cleaned_prices = cleaned_prices.reindex(idx)
            cleaned_volumes = cleaned_volumes.reindex(idx)

        # Be strict about column presence
        if ticker not in cleaned_prices.columns:
            raise KeyError(
                f"{ticker} not in cleaned_prices columns: {list(cleaned_prices.columns)}"
            )
        if ticker not in cleaned_volumes.columns:
            raise KeyError(
                f"{ticker} not in cleaned_volumes columns: {list(cleaned_volumes.columns)}"
            )

        price_series = cleaned_prices[ticker]
        volume_series = cleaned_volumes[ticker]

        pd.DataFrame({"price": price_series, "volume": volume_series}).to_parquet(
            p_path
        )

        prices_dict[ticker] = price_series
        volumes_dict[ticker] = volume_series

    return prices_dict, volumes_dict


def read_processed(ticker, start, end):
    """
    Read processed parquet if available, else return None.
    """
    p_path = _processed_path(ticker, start, end)
    if p_path.exists():
        return pd.read_parquet(p_path)
    return None
