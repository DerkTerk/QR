"""
Clean intraday price/volume data to regular trading hours.

Usage:
    from clean_intraday import clean_to_rth
    prices_rth, volumes_rth = clean_to_rth(prices, volumes)
"""

import pandas as pd


def clean_to_rth(prices, volumes, tz="America/New_York", include_close_bar=True):
    """
    Restrict prices and volumes to NYSE regular trading hours (09:30–16:00).

    Parameters
    ----------
    prices, volumes : pd.DataFrame
        Indexed by datetime (UTC or tz-aware).
    tz : str
        Timezone for filtering (default America/New_York).
    include_close_bar : bool
        If True, include the 16:00 bar; if False, end at 15:59.

    Returns
    -------
    prices_rth, volumes_rth : pd.DataFrame
    """

    def _ensure_tz(df):
        if df.index.tz is None:
            return df.tz_localize("UTC").tz_convert(tz)
        return df.tz_convert(tz)

    prices = _ensure_tz(prices).sort_index()
    volumes = _ensure_tz(volumes).sort_index()

    # Weekdays only
    prices = prices[prices.index.dayofweek < 5]
    volumes = volumes[volumes.index.dayofweek < 5]

    # Filter by time of day
    start_s = "09:30"
    end_s = "16:00"
    inclusive = "both" if include_close_bar else "left"

    prices_rth = prices.between_time(start_s, end_s, inclusive=inclusive)
    volumes_rth = volumes.between_time(start_s, end_s, inclusive=inclusive)

    # Align and drop rows where both price & volume are NaN
    idx = prices_rth.index.union(volumes_rth.index)
    prices_rth = prices_rth.reindex(idx)
    volumes_rth = volumes_rth.reindex(idx)

    mask = ~(prices_rth.isna().all(axis=1) & volumes_rth.isna().all(axis=1))
    return prices_rth.loc[mask], volumes_rth.loc[mask]
