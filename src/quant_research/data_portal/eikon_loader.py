"""
Eikon data loader.

Fetches historical CLOSE and VOLUME via ek.get_timeseries (chunked)
and industry classification via ek.get_data.
"""

import os
import pandas as pd
import eikon as ek
from dotenv import load_dotenv
from datetime import datetime, date, timedelta

# Load .env if present
load_dotenv()
ek.set_app_key(os.getenv("EIKON_APP_KEY"))


def _to_datetime(x):
    """Convert string/date/datetime to datetime.datetime."""
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime.combine(x, datetime.min.time())
    if isinstance(x, timedelta):
        raise TypeError("Use timedelta only for 'end', not 'start'.")
    return datetime.strptime(str(x), "%Y-%m-%d")  # strict ISO format


def _resolve_dates(start, end):
    start_dt = _to_datetime(start)
    if isinstance(end, timedelta):
        end_dt = start_dt + end
    else:
        end_dt = _to_datetime(end)
    if end_dt <= start_dt:
        raise ValueError(f"end_date ({end_dt}) must be after start_date ({start_dt}).")
    return start_dt, end_dt


def fetch_prices_volumes(
    tickers, start, end, interval="minute", fields=None, chunk_days=None
):
    """
    Fetch CLOSE & VOLUME from Eikon with chunking, datetime inputs.

    Parameters
    ----------
    tickers : list[str]
    start   : str | datetime | date
    end     : str | datetime | date | timedelta
    interval: str, default "minute"
    fields  : list[str] or None
    chunk_days : int, default 30 for minute, 3650 for daily

    Returns
    -------
    prices, volumes : pd.DataFrame, pd.DataFrame
    """
    if fields is None:
        fields = ["CLOSE", "VOLUME"]

    start_dt, end_dt = _resolve_dates(start, end)
    if chunk_days is None:
        chunk_days = 30 if interval.lower() == "minute" else 3650

    price_frames, volume_frames = [], []
    step = timedelta(minutes=1) if interval.lower() == "minute" else timedelta(days=1)
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)

        ts = ek.get_timeseries(
            rics=tickers,
            fields=fields,
            start_date=current_start,
            end_date=current_end,
            interval=interval,
            normalize=True,
        )

        # detect instrument column
        inst_col_candidates = [
            c for c in ts.columns if c.lower() not in ("date", "field", "value")
        ]
        if not inst_col_candidates:
            raise KeyError(
                f"Could not find instrument column in returned columns: {list(ts.columns)}"
            )
        inst_col = inst_col_candidates[0]

        p = ts[ts["Field"] == "CLOSE"].pivot(
            index="Date", columns=inst_col, values="Value"
        )
        v = ts[ts["Field"] == "VOLUME"].pivot(
            index="Date", columns=inst_col, values="Value"
        )

        price_frames.append(p)
        volume_frames.append(v)
        current_start = current_end + step

    prices = pd.concat(price_frames, axis=0).sort_index()
    volumes = pd.concat(volume_frames, axis=0).sort_index()

    wanted = [t for t in tickers if t in prices.columns]
    if wanted:
        prices = prices.reindex(columns=wanted)
        volumes = volumes.reindex(columns=wanted)

    return prices, volumes


def fetch_industry_map(tickers, chunk_size=200):
    """
    Fetch TRBC industry classification for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        RICs to fetch.
    chunk_size : int
        Number of tickers per request.

    Returns
    -------
    pd.DataFrame with columns ['symbol', 'industry']
    """
    frames = []
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]
        df, err = ek.get_data(instruments=batch, fields=["TRBC_EconomicSector"])
        if err:
            raise RuntimeError(err)
        frames.append(
            pd.DataFrame(
                {"symbol": df["Instrument"], "industry": df["Economic Sector"]}
            )
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates("symbol")
