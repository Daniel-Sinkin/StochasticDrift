"""
orchestrate_pulls.py

Sequentially runs weekly pulls for 4 months (default: Jan-Apr 2024) for:
  - usdc_weth
  - wbtc_weth

It does NOT merge anything. It just leaves you with weekly parquet files, e.g.:

dataset_out/usdc_weth/weekly/usdc_weth_raw_minimal_fee500_2024-01-01_to_2024-01-08.parquet
dataset_out/wbtc_weth/weekly/wbtc_weth_raw_minimal_fee500_2024-01-01_to_2024-01-08.parquet
...

It is resume-safe: if a weekly output already exists, it skips it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXTRACTOR_SCRIPT = "pull_raw_pair_arbitrum_uniswapv3.py"
DEFAULT_FEE = 500
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
PAIRS = ["usdc_weth", "wbtc_weth"]


def parse_yyyy_mm(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def month_start_end(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    if month == 12:
        end = dt.date(year + 1, 1, 1)
    else:
        end = dt.date(year, month + 1, 1)
    return start, end


def split_into_weeks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    out: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=7), end)
        out.append((cur, nxt))
        cur = nxt
    return out


def fmt_date(d: dt.date) -> str:
    return d.isoformat()


@dataclass(frozen=True)
class RunSpec:
    pair: str
    start: dt.date
    end: dt.date
    fee: int
    chunk_size: int
    out_dir: Path


def expected_output_path(spec: RunSpec) -> Path:
    start_s = fmt_date(spec.start)
    end_s = fmt_date(spec.end)
    name = f"{spec.pair}_raw_minimal_fee{spec.fee}_{start_s}_to_{end_s}.parquet"
    return spec.out_dir / name


def run_extractor(spec: RunSpec) -> None:
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = expected_output_path(spec)
    if out_path.exists():
        print(f"\n=== SKIP (exists) ===\n{out_path}")
        return

    start_s = fmt_date(spec.start)
    end_s = fmt_date(spec.end)

    cmd = [
        sys.executable,
        EXTRACTOR_SCRIPT,
        "--pair",
        spec.pair,
        "--start",
        start_s,
        "--end",
        end_s,
        "--fee",
        str(spec.fee),
        "--chunk-size",
        str(spec.chunk_size),
        "--out-dir",
        spec.out_dir.as_posix(),
    ]

    print("\n=== RUN ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.exists():
        raise FileNotFoundError(f"Expected output not found after run: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="*", default=DEFAULT_MONTHS)
    parser.add_argument("--root-out", default="dataset_out")
    parser.add_argument("--fee", type=int, default=DEFAULT_FEE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    extractor_path = Path(EXTRACTOR_SCRIPT).resolve()
    print("Extractor script:", extractor_path)
    if not extractor_path.exists():
        raise FileNotFoundError(f"Extractor script not found: {extractor_path}")

    root_out = Path(args.root_out).resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        weekly_dir = root_out / pair / "weekly"
        weekly_dir.mkdir(parents=True, exist_ok=True)

        for ym in args.months:
            y, m = parse_yyyy_mm(ym)
            m_start, m_end = month_start_end(y, m)
            weeks = split_into_weeks(m_start, m_end)

            print("\n==============================")
            print(f"PAIR={pair} MONTH={ym} ({m_start} to {m_end}) weeks={len(weeks)}")
            print("==============================")

            for w_start, w_end in weeks:
                spec = RunSpec(
                    pair=pair,
                    start=w_start,
                    end=w_end,
                    fee=args.fee,
                    chunk_size=args.chunk_size,
                    out_dir=weekly_dir,
                )
                run_extractor(spec)

    print("\nAll done.")
    print(f"Weekly outputs are in: {root_out}/<pair>/weekly/")


if __name__ == "__main__":
    main()
