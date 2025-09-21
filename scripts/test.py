from pathlib import Path

from quant_research.data_portal.hl_ctxs_to_parquet import split_asset_ctxs_by_coin_month

RAW_DIR = Path("data/hl_asset_ctxs_raw")
PARQUET_DIR = Path("data/hl_assets_ctxs_parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
COIN_DIR = Path("data/hl_assets_ctxs_by_coin")
COIN_DIR.mkdir(parents=True, exist_ok=True)

summary = split_asset_ctxs_by_coin_month(
    in_dir=PARQUET_DIR,
    out_dir=COIN_DIR,
    file_glob="*.parquet",  # matches 20240101.parquet etc.
    compression="zstd",
    row_group_size=1_000_000,
    overwrite=True,
)

print("Rows written per (coin, YYYY-MM):")
for (coin, month), n in sorted(summary.items()):
    print(coin, month, n)
