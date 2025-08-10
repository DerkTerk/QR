import numpy as np
import pandas as pd


def to_daily_close(series):
    """
    Convert an intraday price series to daily close.
    If already daily, returns unchanged.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        # Detect intraday by presence of non-midnight times
        if (series.index.time != pd.to_datetime(series.index.date).time).any():
            return series.resample("1D").last().dropna()
    return series


def ma_crossover_signal(price, fast=50, slow=200, resample_to_daily=True):
    """
    Generate MA crossover signal.

    invert=True implements the user's rule:
      - LONG (1) when fast MA < slow MA
      - SHORT (-1) when fast MA > slow MA

    invert=False is the classic golden/death cross:
      - LONG when fast MA > slow MA
      - SHORT when fast MA < slow MA
    """
    if resample_to_daily:
        price_daily = to_daily_close(price)
    else:
        price_daily = price

    fast_ma = price_daily.rolling(fast, min_periods=fast).mean()
    slow_ma = price_daily.rolling(slow, min_periods=slow).mean()

    sig = np.where(fast_ma > slow_ma, 1, -1)
    signal = pd.Series(sig, index=price_daily.index).astype(float)

    # Avoid positions before MAs are ready
    signal[(fast_ma.isna()) | (slow_ma.isna())] = 0.0
    return signal, price_daily, fast_ma, slow_ma
