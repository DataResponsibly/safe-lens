# Stress Testing (Concurrent Users)

This folder contains a browser-level stress test for Safe-Lens using Playwright.

## Why this test

It simulates real concurrent users on the deployed UI and reports:

- Success/failure count
- Throughput
- Latency (avg/p50/p95/p99)
- Sample failure reasons

## Setup

From the project root:

```bash
python3 -m pip install playwright
python3 -m playwright install chromium
```

## Run

```bash
python3 tests/stress_test_playwright.py \
  --url "https://safe-lens.users.hsrn.nyu.edu/" \
  --users 20 \
  --iterations 5 \
  --timeout-ms 120000 \
  --ramp-up-ms 200
```

## Useful presets

Light sanity check:

```bash
python3 tests/stress_test_playwright.py --users 5 --iterations 2
```

Heavier burst:

```bash
python3 tests/stress_test_playwright.py --users 50 --iterations 3 --ramp-up-ms 50
```

## Notes

- This is browser-based load, so local CPU/RAM can become the bottleneck first.
- For very large loads, consider running this on a dedicated machine or splitting across multiple runners.
- If your backend streams slowly, increase `--timeout-ms`.
