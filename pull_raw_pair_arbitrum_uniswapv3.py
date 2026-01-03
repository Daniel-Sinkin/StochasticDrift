import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd
from web3 import Web3


def get_factory_abi() -> list[dict]:
    return [
        {
            "name": "getPool",
            "type": "function",
            "stateMutability": "view",
            "inputs": [
                {"name": "tokenA", "type": "address"},
                {"name": "tokenB", "type": "address"},
                {"name": "fee", "type": "uint24"},
            ],
            "outputs": [{"name": "pool", "type": "address"}],
        }
    ]


POOL_ABI = [
    {
        "name": "token0",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "address"}],
    },
    {
        "name": "token1",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "address"}],
    },
]

ERC20_ABI = [
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "uint8"}],
    },
    {
        "name": "symbol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"type": "string"}],
    },
]


@dataclass(frozen=True)
class Addresses:
    # Arbitrum One
    WETH: str = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
    USDC: str = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
    WBTC: str = "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f"
    UNISWAP_V3_FACTORY: str = "0x1F98431c8aD98523631AE4a59f267346ea31F984"


SWAP_EVENT_SIG = "Swap(address,address,int256,int256,uint160,uint128,int24)"


def get_arb_rpc_url() -> str:
    with open("api_keys.json", "r", encoding="utf8") as file:
        d = json.load(file)
    return d["arb_rpc_url"]


def keccak_topic0(w3: Web3, event_sig: str) -> str:
    return "0x" + w3.keccak(text=event_sig).hex()


def utc_ts(dtime: dt.datetime) -> int:
    if dtime.tzinfo is None:
        dtime = dtime.replace(tzinfo=dt.timezone.utc)
    return int(dtime.timestamp())


def parse_yyyy_mm_dd(s: str) -> dt.datetime:
    y, m, d = s.split("-")
    return dt.datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=dt.timezone.utc)


def find_block_by_timestamp(w3: Web3, target_ts: int, *, side: str) -> int:
    """
    side="left": greatest block with timestamp <= target_ts
    side="right": smallest block with timestamp >= target_ts
    """
    latest = w3.eth.get_block("latest")
    hi = int(latest["number"])
    lo = 0

    if target_ts <= int(w3.eth.get_block(lo)["timestamp"]):
        return lo
    if target_ts >= int(latest["timestamp"]):
        return hi

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ts = int(w3.eth.get_block(mid)["timestamp"])
        if ts <= target_ts:
            lo = mid
        else:
            hi = mid

    if side == "left":
        return lo
    if side == "right":
        return hi
    raise ValueError("side must be 'left' or 'right'")


def find_pool_for_fee(w3: Web3, token_a: str, token_b: str, fee: int) -> str:
    factory = w3.eth.contract(
        address=w3.to_checksum_address(Addresses.UNISWAP_V3_FACTORY),
        abi=get_factory_abi(),
    )
    return factory.functions.getPool(
        w3.to_checksum_address(token_a),
        w3.to_checksum_address(token_b),
        fee,
    ).call()


def get_token_meta(w3: Web3, token: str) -> tuple[int, str]:
    c = w3.eth.contract(address=w3.to_checksum_address(token), abi=ERC20_ABI)
    dec = int(c.functions.decimals().call())
    sym = str(c.functions.symbol().call())
    return dec, sym


def get_pool_tokens_and_decimals(w3: Web3, pool: str) -> dict[str, Any]:
    c = w3.eth.contract(address=w3.to_checksum_address(pool), abi=POOL_ABI)
    token0 = w3.to_checksum_address(c.functions.token0().call())
    token1 = w3.to_checksum_address(c.functions.token1().call())
    dec0, sym0 = get_token_meta(w3, token0)
    dec1, sym1 = get_token_meta(w3, token1)
    return {
        "token0": token0,
        "token1": token1,
        "dec0": dec0,
        "dec1": dec1,
        "sym0": sym0,
        "sym1": sym1,
    }


def decode_swap_sqrt_price_x96(w3: Web3, log: dict) -> int:
    data_field = log["data"]
    if isinstance(data_field, (bytes, bytearray)):
        data = bytes(data_field)
    else:
        try:
            data = bytes(data_field)
        except TypeError:
            data = bytes.fromhex(str(data_field)[2:])

    types = ["int256", "int256", "uint160", "uint128", "int24"]
    _, _, sqrtPriceX96, _, _ = w3.codec.decode(types, data)
    return int(sqrtPriceX96)


def sqrtprice_to_price_token1_per_token0(sqrt_price_x96: int) -> float:
    x = sqrt_price_x96 / float(2**96)
    return x * x


def adjust_for_decimals(price_token1_per_token0: float, dec0: int, dec1: int) -> float:
    return price_token1_per_token0 * (10.0 ** (dec0 - dec1))


def price_quote_per_base_from_pool_price(
    *,
    price_token1_per_token0_human: float,
    token0: str,
    token1: str,
    base: str,
    quote: str,
) -> float:
    token0 = Web3.to_checksum_address(token0)
    token1 = Web3.to_checksum_address(token1)
    base = Web3.to_checksum_address(base)
    quote = Web3.to_checksum_address(quote)

    if base == token0 and quote == token1:
        return price_token1_per_token0_human
    if base == token1 and quote == token0:
        return 1.0 / price_token1_per_token0_human

    raise RuntimeError("Unexpected token ordering for base/quote vs pool token0/token1")


def _looks_like_too_many_logs_error(e: Exception) -> bool:
    msg = str(e)
    # We’ve seen:
    # "Log response size exceeded. ... cap of 10K logs in the response."
    return (
        ("Log response size exceeded" in msg)
        or ("cap of 10K logs" in msg)
        or ("response size exceeded" in msg)
    )


def get_logs_adaptive(
    w3: Web3,
    *,
    address: str,
    topic0: str,
    from_block: int,
    to_block: int,
    max_block_span: int,
    min_block_span: int = 1000,
) -> list[tuple[int, int, list[dict]]]:
    """
    Returns a list of (b0, b1, logs) that together cover [from_block, to_block].

    Strategy:
    - Attempt spans up to max_block_span.
    - If RPC errors due to log response size, halve the span and retry.
    - If span becomes too small, we still proceed (down to min_block_span), otherwise error.

    This makes the pipeline robust to Alchemy's log-count caps.
    """
    out: list[tuple[int, int, list[dict]]] = []
    cur = from_block

    while cur <= to_block:
        span = min(max_block_span, to_block - cur + 1)
        b0 = cur
        b1 = min(cur + span - 1, to_block)

        while True:
            try:
                logs = w3.eth.get_logs(
                    {
                        "fromBlock": b0,
                        "toBlock": b1,
                        "address": w3.to_checksum_address(address),
                        "topics": [topic0],
                    }
                )
                out.append((b0, b1, logs))
                cur = b1 + 1
                break
            except Exception as e:
                if _looks_like_too_many_logs_error(e):
                    old_span = b1 - b0 + 1
                    if old_span <= min_block_span:
                        raise RuntimeError(
                            f"eth_getLogs keeps exceeding log cap even at span={old_span} "
                            f"for window {b0}-{b1}. Consider lowering min_block_span."
                        ) from e
                    new_span = max(min_block_span, old_span // 2)
                    b1 = b0 + new_span - 1
                    print(
                        f"[adaptive] too many logs in {b0}-{b0 + old_span - 1} "
                        f"-> retry with span={new_span} ({b0}-{b1})"
                    )
                    continue
                raise

    return out


def pull_raw_swaps_minimal(
    w3: Web3,
    *,
    pool: str,
    base_token: str,
    quote_token: str,
    start_block: int,
    end_block: int,
    chunk_size: int = 50_000,
    min_span: int = 2_000,
) -> pd.DataFrame:
    """
    Minimal dataset:
      - timestamp (int64)
      - block_number (int64)
      - log_index (int64)
      - log_price (float64)

    Uses adaptive eth_getLogs to respect provider caps (10k logs response).
    """
    topic0 = keccak_topic0(w3, SWAP_EVENT_SIG)
    ts_cache: dict[int, int] = {}

    meta = get_pool_tokens_and_decimals(w3, pool)
    token0 = meta["token0"]
    token1 = meta["token1"]
    dec0 = meta["dec0"]
    dec1 = meta["dec1"]
    print(f"pool tokens: {meta['sym0']} ({dec0}) / {meta['sym1']} ({dec1})")

    rows: list[dict[str, Any]] = []

    total_blocks = end_block - start_block + 1
    est_chunks = (total_blocks + chunk_size - 1) // chunk_size
    t0 = perf_counter()

    # We iterate over coarse chunks, but inside each we may split further.
    coarse_idx = 0
    cur = start_block

    while cur <= end_block:
        coarse_idx += 1
        coarse_b0 = cur
        coarse_b1 = min(cur + chunk_size - 1, end_block)

        segments = get_logs_adaptive(
            w3,
            address=pool,
            topic0=topic0,
            from_block=coarse_b0,
            to_block=coarse_b1,
            max_block_span=coarse_b1 - coarse_b0 + 1,
            min_block_span=min_span,
        )

        swaps_in_coarse = 0
        for seg_b0, seg_b1, logs in segments:
            swaps_in_coarse += len(logs)

            for lg in logs:
                bn = int(lg["blockNumber"])
                if bn not in ts_cache:
                    ts_cache[bn] = int(w3.eth.get_block(bn)["timestamp"])
                ts = ts_cache[bn]

                sqrt_price_x96 = decode_swap_sqrt_price_x96(w3, lg)
                raw_price = sqrtprice_to_price_token1_per_token0(sqrt_price_x96)
                human_price = adjust_for_decimals(raw_price, dec0, dec1)
                quote_per_base = price_quote_per_base_from_pool_price(
                    price_token1_per_token0_human=human_price,
                    token0=token0,
                    token1=token1,
                    base=base_token,
                    quote=quote_token,
                )

                rows.append(
                    {
                        "timestamp": int(ts),
                        "block_number": bn,
                        "log_index": int(lg["logIndex"]),
                        "log_price": float(math.log(quote_per_base)),
                    }
                )

        elapsed = perf_counter() - t0
        print(
            f"[coarse {coarse_idx:>4}/{est_chunks}] blocks {coarse_b0}-{coarse_b1} | "
            f"swaps={swaps_in_coarse:>7} | rows={len(rows):>9} | elapsed={elapsed:>7.1f}s"
        )

        cur = coarse_b1 + 1

    df = pd.DataFrame(rows)
    df.sort_values(["timestamp", "block_number", "log_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["timestamp"] = df["timestamp"].astype("int64")
    df["block_number"] = df["block_number"].astype("int64")
    df["log_index"] = df["log_index"].astype("int64")
    df["log_price"] = df["log_price"].astype("float64")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["usdc_weth", "wbtc_weth"], required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, exclusive)")
    parser.add_argument("--fee", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--min-span",
        type=int,
        default=2_000,
        help="Minimum block span during adaptive splitting",
    )
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w3 = Web3(Web3.HTTPProvider(get_arb_rpc_url()))
    print("connected:", w3.is_connected(), "latest block:", w3.eth.block_number)

    if args.pair == "usdc_weth":
        base = Addresses.WETH
        quote = Addresses.USDC
        pair_name = "USDC-WETH"
    else:
        base = Addresses.WETH
        quote = Addresses.WBTC
        pair_name = "WBTC-WETH"

    print("\nPair:", pair_name)

    pool = find_pool_for_fee(w3, base, quote, args.fee)
    if int(pool, 16) == 0:
        raise RuntimeError("Pool not found. Check fee tier / token addresses.")
    print("Using fee", args.fee, "pool", pool)

    start_dt = parse_yyyy_mm_dd(args.start)
    end_dt = parse_yyyy_mm_dd(args.end)

    start_block = find_block_by_timestamp(w3, utc_ts(start_dt), side="right")
    end_block = find_block_by_timestamp(w3, utc_ts(end_dt), side="left")
    print(
        "start_block:",
        start_block,
        "end_block:",
        end_block,
        "block_span:",
        end_block - start_block + 1,
    )

    print("\nPulling raw swaps (minimal schema, adaptive logs)...")
    df = pull_raw_swaps_minimal(
        w3,
        pool=pool,
        base_token=base,
        quote_token=quote,
        start_block=start_block,
        end_block=end_block,
        chunk_size=args.chunk_size,
        min_span=args.min_span,
    )

    out_prefix = f"{args.pair}_raw_minimal_fee{args.fee}_{args.start}_to_{args.end}"
    out_parquet = out_dir / (out_prefix + ".parquet")

    print("\nWriting:", out_parquet.as_posix())
    df.to_parquet(out_parquet, index=False)

    print("\nDone.")
    print("rows:", len(df))
    if len(df) > 0:
        print(
            "first ts:",
            int(df["timestamp"].iloc[0]),
            "last ts:",
            int(df["timestamp"].iloc[-1]),
        )


if __name__ == "__main__":
    main()
