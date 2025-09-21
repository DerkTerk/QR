import re
from collections import OrderedDict
from pathlib import Path

import lz4.frame
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

RAW_DIR = Path("data/hl_asset_ctxs_raw")
PARQUET_DIR = Path("data/hl_assets_ctxs_parquet")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
COIN_DIR = Path("data/hl_assets_ctxs_by_coin")
COIN_DIR.mkdir(parents=True, exist_ok=True)


def convert_parquets():
    for f in sorted(RAW_DIR.glob("*.csv.lz4")):
        # robust date extraction
        m = re.match(r"^(\d{8})\.csv\.lz4$", f.name)
        if not m:
            print("skip (unexpected name):", f.name)
            continue
        date_str = m.group(1)

        parquet_path = PARQUET_DIR / f"{date_str}.parquet"
        if parquet_path.exists():
            print("skip", parquet_path)
            continue

        with lz4.frame.open(f, "rb") as fin:
            df = pd.read_csv(fin)

        df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
        df.to_parquet(parquet_path, index=False)
        print("wrote", parquet_path)


# ---- Default target schema (adjust types if your source differs) ----
DEFAULT_SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("ns")),
        ("coin", pa.string()),
        ("funding", pa.float32()),  # flip to float64 if you prefer
        ("open_interest", pa.float32()),
        ("prev_day_px", pa.float32()),
        ("day_ntl_vlm", pa.float32()),
        ("premium", pa.float32()),
        ("oracle_px", pa.float32()),
        ("mark_px", pa.float32()),
        ("mid_px", pa.float32()),
        ("impact_bid_px", pa.float32()),
        ("impact_ask_px", pa.float32()),
        ("date", pa.date32()),
    ]
)


def _parse_timestamp_ns_from_str(arr):
    """
    Parse common ISO 8601 variants to timestamp[ns].
    Handles trailing 'Z' and ±HH:MM offsets by stripping them first,
    then tries multiple formats and coalesces.
    """
    s = arr
    # Ensure string
    if not pa.types.is_string(s.type):
        return s  # not string; let caller handle
    # Strip trailing 'Z' and numeric tz offsets like +00:00 or -05:30
    s = pc.replace_substring_regex(s, pattern=r"Z$", replacement="")
    s = pc.replace_substring_regex(s, pattern=r"[+-]\d{2}:\d{2}$", replacement="")

    # Try multiple layouts (seconds / no-seconds, 'T' or space)
    ts1 = pc.strptime(s, format="%Y-%m-%dT%H:%M:%S", unit="ns", error_is_null=True)
    ts2 = pc.strptime(s, format="%Y-%m-%d %H:%M:%S", unit="ns", error_is_null=True)
    ts3 = pc.strptime(s, format="%Y-%m-%dT%H:%M", unit="ns", error_is_null=True)
    ts4 = pc.strptime(s, format="%Y-%m-%d %H:%M", unit="ns", error_is_null=True)
    return pc.coalesce(ts1, ts2, ts3, ts4)  # first non-null per row


def _parse_date32_from_str(arr):
    """Parse YYYY-MM-DD (and tolerant ISO variants) to date32."""
    s = arr
    if not pa.types.is_string(s.type):
        return s
    # If there's time info, strip to date first
    s_only = pc.replace_substring_regex(s, pattern=r"[T ].*$", replacement="")
    d1 = pc.strptime(s_only, format="%Y-%m-%d", unit="ns", error_is_null=True)
    # Convert timestamp[ns] -> date32
    return pc.cast(d1, pa.date32())


# --- MODIFY your existing _cast_to_schema --------------------------------


def _cast_to_schema(tbl, target_schema):
    cols = []
    names = set(tbl.schema.names)

    for field in target_schema:
        if field.name in names:
            col = tbl[field.name]

            # Special-case time/date parsing from strings
            if pa.types.is_timestamp(field.type):
                # If col is string or tz-aware timestamp, normalize to timestamp[ns]
                if pa.types.is_string(col.type):
                    col = _parse_timestamp_ns_from_str(col)
                elif isinstance(col.type, pa.TimestampType) and col.type.tz is not None:
                    # Drop timezone to naive ns
                    col = pc.cast(col, pa.timestamp("ns"))
                # Final cast (no-op if already timestamp[ns])
                if not col.type.equals(field.type):
                    col = pc.cast(col, field.type)

            elif pa.types.is_date(field.type):
                if pa.types.is_string(col.type):
                    col = _parse_date32_from_str(col)
                if not col.type.equals(field.type):
                    col = pc.cast(col, field.type)

            else:
                # Generic numeric/string casts
                if not col.type.equals(field.type):
                    try:
                        col = pc.cast(col, field.type)
                    except Exception:
                        valid = pc.is_valid(col)
                        col = pc.cast(
                            pc.if_else(valid, col, pa.nulls(len(col), type=field.type)),
                            field.type,
                        )

            cols.append(col)
        else:
            cols.append(pa.nulls(tbl.num_rows, type=field.type))

    return pa.Table.from_arrays(cols, schema=target_schema)


def _month_str(arr):
    ts = arr if pa.types.is_timestamp(arr.type) else pc.cast(arr, pa.timestamp("ns"))
    return pc.strftime(ts, format="%Y-%m")


def split_asset_ctxs_by_coin_month(
    in_dir,
    out_dir,
    file_glob="*.parquet",
    compression="zstd",
    row_group_size=1_000_000,
    target_schema=DEFAULT_SCHEMA,
    overwrite=True,
    max_open_writers=64,
):
    in_dir = Path(in_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand the glob ourselves (PyArrow doesn't)
    files = sorted((in_dir).glob(file_glob))
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir.resolve()}")
    if not files:
        raise FileNotFoundError(
            "No Parquet files matched.\n"
            f"- Working dir: {Path.cwd().resolve()}\n"
            f"- Looked for:  {in_dir / file_glob}\n"
            f"- Tip: verify the folder name and try file_glob='**/*.parquet' if nested."
        )

    dataset = ds.dataset([str(p) for p in files], format="parquet")

    #  FILTER IN THE SCAN (Expression API)
    filt = (ds.field("coin").is_valid()) & (ds.field("time").is_valid())
    scanner = dataset.scanner(columns=target_schema.names, filter=filt)

    # LRU cache: (coin, 'YYYY-MM') -> ParquetWriter
    writers = OrderedDict()
    counts = {}

    def _writer_for(coin, yyyymm):
        key = (coin, yyyymm)
        # touch if present
        if key in writers:
            writers.move_to_end(key)
            return writers[key]

        coin_dir = out_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        out_path = coin_dir / f"{yyyymm}.parquet"

        if overwrite and out_path.exists():
            out_path.unlink()

        # If at capacity, close least-recently-used writer
        if len(writers) >= max_open_writers:
            old_key, old_writer = writers.popitem(last=False)  # LRU
            try:
                old_writer.close()
            except Exception:
                pass  # keep going

        w = pq.ParquetWriter(
            where=str(out_path),
            schema=target_schema,
            compression=compression,
            use_dictionary=True,
        )
        writers[key] = w
        return w

    try:
        for rec_batch in scanner.to_batches():
            tbl = pa.Table.from_batches([rec_batch])

            # Normalize schema (handles drift / missing cols)
            tbl = _cast_to_schema(tbl, target_schema)
            if tbl.num_rows == 0:
                continue

            # Group by coin first
            coins = pc.unique(tbl["coin"]).to_pylist()
            for coin in coins:
                c_tbl = tbl.filter(pc.equal(tbl["coin"], pa.scalar(coin)))
                if c_tbl.num_rows == 0:
                    continue

                # Month extraction — prefer 'date'; fall back to 'time' if needed
                base_date = c_tbl["date"]
                if (
                    not pa.types.is_date32(base_date.type)
                    and not pc.any(pc.is_valid(base_date)).as_py()
                ):
                    base_date = c_tbl[
                        "time"
                    ]  # in case date missing; still yields YYYY-MM
                months = _month_str(base_date)
                uniq_months = pc.unique(months).to_pylist()

                for m in uniq_months:
                    m_mask = pc.equal(months, pa.scalar(m))
                    cm_tbl = c_tbl.filter(m_mask)
                    if cm_tbl.num_rows == 0:
                        continue

                    writer = _writer_for(coin, m)
                    if cm_tbl.num_rows > row_group_size:
                        for start in range(0, cm_tbl.num_rows, row_group_size):
                            writer.write_table(cm_tbl.slice(start, row_group_size))
                    else:
                        writer.write_table(cm_tbl)

                    counts[(coin, m)] = counts.get((coin, m), 0) + cm_tbl.num_rows
    finally:
        # Close any remaining open writers
        for w in list(writers.values()):
            try:
                w.close()
            except Exception:
                pass

    return counts


# Optional convenience: per-coin single file (if you ever need it)
def split_asset_ctxs_by_coin_single(
    in_dir,
    out_dir,
    file_glob="*.parquet",
    compression="zstd",
    target_schema=DEFAULT_SCHEMA,
    required_non_null=("coin", "time"),
    overwrite=True,
):
    """
    Write one big Parquet per coin to `out_dir/COIN.parquet`.
    Not recommended for very large coins; prefer monthly partitioning.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(in_dir / file_glob), format="parquet")
    scanner = ds.Scanner(dataset=dataset, columns=target_schema.names)

    tables_by_coin = {}

    for rec_batch in scanner.to_batches():
        tbl = pa.Table.from_batches([rec_batch])

        mask = pc.scalar(True)
        for col in required_non_null:
            mask = pc.and_(mask, pc.is_valid(tbl[col]))
        tbl = tbl.filter(mask)
        if tbl.num_rows == 0:
            continue

        tbl = _cast_to_schema(tbl, target_schema)

        coins = pc.unique(tbl["coin"]).to_pylist()
        for coin in coins:
            c_tbl = tbl.filter(pc.equal(tbl["coin"], pa.scalar(coin)))
            if c_tbl.num_rows == 0:
                continue
            tables_by_coin.setdefault(coin, []).append(c_tbl)

    for coin, parts in tables_by_coin.items():
        coin_tbl = pa.concat_tables(parts, promote=True)
        out_path = out_dir / f"{coin}.parquet"
        if overwrite and out_path.exists():
            out_path.unlink()
        pq.write_table(coin_tbl, out_path, compression=compression)
