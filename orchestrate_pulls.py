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
DEFAULT_MIN_SPAN = 2_000

# For your “start with 4 months” plan
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
    """
    Split [start, end) into 7-day chunks. Last chunk may be shorter.
    """
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
    min_span: int
    out_dir: Path


def expected_output_path(spec: RunSpec) -> Path:
    start_s = fmt_date(spec.start)
    end_s = fmt_date(spec.end)
    name = f"{spec.pair}_raw_minimal_fee{spec.fee}_{start_s}_to_{end_s}.parquet"
    return spec.out_dir / name


def run_extractor_if_needed(spec: RunSpec) -> None:
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
        "--min-span",
        str(spec.min_span),
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
    parser.add_argument("--min-span", type=int, default=DEFAULT_MIN_SPAN)
    parser.add_argument(
        "--pairs", nargs="*", default=PAIRS, help="Override which pairs to run"
    )
    args = parser.parse_args()

    extractor_path = Path(EXTRACTOR_SCRIPT).resolve()
    print("Extractor script:", extractor_path)
    if not extractor_path.exists():
        raise FileNotFoundError(f"Extractor script not found: {extractor_path}")

    root_out = Path(args.root_out).resolve()
    root_out.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
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
                    min_span=args.min_span,
                    out_dir=weekly_dir,
                )
                run_extractor_if_needed(spec)

    print("\nAll done.")
    print(f"Weekly outputs are in: {root_out}/<pair>/weekly/")


if __name__ == "__main__":
    main()
