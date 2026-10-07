"""Local in-process latency benchmark of POST /api/v1/transactions/analyze (synthetic data).

Run from mfs-guard-ml/: .venv/Scripts/python scripts/benchmark_api.py
"""
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

from fastapi.testclient import TestClient  # noqa: E402
from backend import demo_scenarios as demo  # noqa: E402
from backend.app.main import Analyzer, create_app  # noqa: E402

SEQ, CONC, THREADS = 1000, 200, 8


def make_tx(i, rng, t0):
    users = 200
    return {'transaction_id': f'B{i:06d}', 'user_id': f'BU{rng.integers(users)}',
            'receiver_id': f'BU{rng.integers(users)}', 'device_id': f'BD{rng.integers(users + 20)}',
            'timestamp': (t0 + timedelta(seconds=30 * i)).isoformat(),
            'transaction_type': str(rng.choice(['SEND_MONEY', 'CASH_OUT', 'PAYMENT', 'MOBILE_RECHARGE'])),
            'channel': 'APP', 'amount': float(rng.lognormal(7.3, .7)), 'balance_before': 100000.0,
            'failed_pin_attempts': int(rng.poisson(.1)), 'otp_resend_count': 0}


def stats(lat, errors, n):
    a = np.asarray(lat)
    return {'n': n, 'p50': np.percentile(a, 50), 'p95': np.percentile(a, 95), 'p99': np.percentile(a, 99),
            'mean': a.mean(), 'error_rate': errors / n}


def main():
    client = TestClient(create_app(Analyzer(log_dir=Path(tempfile.mkdtemp()))))
    for fn in demo.SCENARIOS.values():
        hist, _ = fn()
        client.post('/api/v1/history/ingest', json={'transactions': hist}).raise_for_status()
    warm = len(client.app.state.analyzer.features.history)
    rng = np.random.default_rng(7)
    t0 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    lat, errors = [], 0
    started = time.perf_counter()
    for i in range(SEQ):
        s = time.perf_counter()
        r = client.post('/api/v1/transactions/analyze', json=make_tx(i, rng, t0))
        lat.append((time.perf_counter() - s) * 1000)
        errors += r.status_code != 200
    seq = stats(lat, errors, SEQ)
    seq['throughput'] = SEQ / (time.perf_counter() - started)

    txs = [make_tx(SEQ + i, rng, t0) for i in range(CONC)]
    clat, cerr = [], 0

    def one(tx):
        s = time.perf_counter()
        r = client.post('/api/v1/transactions/analyze', json=tx)
        return (time.perf_counter() - s) * 1000, r.status_code

    started = time.perf_counter()
    with ThreadPoolExecutor(THREADS) as ex:
        for ms, code in ex.map(one, txs):
            clat.append(ms)
            cerr += code != 200
    conc = stats(clat, cerr, CONC)
    conc['throughput'] = CONC / (time.perf_counter() - started)
    final_hist = len(client.app.state.analyzer.features.history)

    row = lambda name, s: (f"| {name} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | "
                           f"{s['mean']:.1f} | {s['throughput']:.1f} | {s['error_rate']:.2%} |")
    report = f"""# Final integration benchmark — scoring API

**Scope:** local, in-process (FastAPI `TestClient`, no network hop), synthetic transactions,
single machine (Windows dev box), generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.
Not a production load test.

Each request runs the full path: request validation -> feature computation by replaying the
in-memory ledger through `src.feature_engineering.engineer` -> rules -> LightGBM + sigmoid
calibration -> SHAP top factors -> anomaly layer -> decision -> audit log append -> commit to history.

History warm-up: {warm} demo rows; ledger grew to {final_hist} rows by the end.

| Run | Requests | P50 ms | P95 ms | P99 ms | Mean ms | Req/s | Error rate |
|---|---|---|---|---|---|---|---|
{row('Sequential', seq)}
{row(f'Concurrent ({THREADS} threads)', conc)}

**Notes / limitations**
- Feature computation is O(history length) per request (full replay of the ledger, capped at
  20,000 rows), so latency grows with ledger size; an incremental state store would make it O(1).
- Scoring is serialized by a lock so history is committed in order; concurrent requests therefore
  queue, which is reflected in the concurrent latencies (throughput does not scale with threads).
"""
    out = ROOT / 'reports/final_integration_benchmark.md'
    out.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
