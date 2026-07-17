import argparse
import asyncio
import json
import time
from statistics import mean

import httpx

# Stand-in prompts. Real trace replay (arrival timeline + token counts) comes later.
DEFAULT_PROMPTS = [
    "Explain what a GPU is in one sentence.",
    "Write a haiku about servers.",
    "What is the capital of France?",
    "Summarize why the sky is blue.",
]


def pct(values, p):
    """Return the p-th percentile of a list (linear interpolation)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt(x, d=3):
    return f"{x:.{d}f}" if x is not None else "n/a"


async def one_request(client, url, prompt, max_new_tokens):
    """Send one streaming request; measure client-side TTFT and per-token gaps."""
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": 0}
    t0 = time.perf_counter()
    ttft = None
    token_times = []
    completion_tokens = 0
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                now = time.perf_counter()
                if event.get("type") == "token":
                    if ttft is None:
                        ttft = now - t0          # first token -> TTFT
                    token_times.append(now)
                    completion_tokens += 1
                elif event.get("type") == "done":
                    completion_tokens = event.get("completion_tokens", completion_tokens)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    total = time.perf_counter() - t0
    tpots = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    return {
        "ok": True,
        "ttft": ttft,
        "tpot": mean(tpots) if tpots else None,   # this request's avg TPOT
        "latency": total,
        "completion_tokens": completion_tokens,
    }


async def run(url, num_requests, concurrency, max_new_tokens):
    sem = asyncio.Semaphore(concurrency)   # cap how many run at once

    async with httpx.AsyncClient(timeout=120) as client:
        async def worker(i):
            async with sem:
                prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
                return await one_request(client, url, prompt, max_new_tokens)

        wall0 = time.perf_counter()
        results = await asyncio.gather(*[worker(i) for i in range(num_requests)])
        wall = time.perf_counter() - wall0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    total_tokens = sum(r["completion_tokens"] for r in ok)

    print("\n==== Benchmark results ====")
    print(f"requests       : {num_requests} (concurrency {concurrency})")
    print(f"success / fail : {len(ok)} / {len(fail)}")
    print(f"wall time      : {wall:.2f}s")
    print(f"throughput     : {fmt(total_tokens / wall, 1)} tok/s, {fmt(len(ok) / wall, 2)} req/s")
    print(f"TTFT (s)  p50 {fmt(pct(ttfts,50))}  p95 {fmt(pct(ttfts,95))}  p99 {fmt(pct(ttfts,99))}")
    print(f"TPOT (s)  p50 {fmt(pct(tpots,50),4)}  p95 {fmt(pct(tpots,95),4)}  p99 {fmt(pct(tpots,99),4)}")

    summary = {
        "num_requests": num_requests, "concurrency": concurrency,
        "success": len(ok), "fail": len(fail), "wall_s": round(wall, 3),
        "tokens_per_s": total_tokens / wall if wall else None,
        "req_per_s": len(ok) / wall if wall else None,
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95), "ttft_p99": pct(ttfts, 99),
        "tpot_p50": pct(tpots, 50), "tpot_p95": pct(tpots, 95), "tpot_p99": pct(tpots, 99),
    }
    fname = f"benchmarks/results_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/generate/stream")
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.num_requests, args.concurrency, args.max_new_tokens))