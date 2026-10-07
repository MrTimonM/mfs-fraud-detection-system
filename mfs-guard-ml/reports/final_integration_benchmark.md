# Final integration benchmark — scoring API

**Scope:** local, in-process (FastAPI `TestClient`, no network hop), synthetic transactions,
single machine (Windows dev box), generated 2026-10-07T05:25:55+00:00.
Not a production load test.

Each request runs the full path: request validation -> feature computation by replaying the
in-memory ledger through `src.feature_engineering.engineer` -> rules -> LightGBM + sigmoid
calibration -> SHAP top factors -> anomaly layer -> decision -> audit log append -> commit to history.

History warm-up: 118 demo rows; ledger grew to 1318 rows by the end.

| Run | Requests | P50 ms | P95 ms | P99 ms | Mean ms | Req/s | Error rate |
|---|---|---|---|---|---|---|---|
| Sequential | 1000 | 199.2 | 364.0 | 561.0 | 211.6 | 4.7 | 0.00% |
| Concurrent (8 threads) | 200 | 2487.0 | 2690.0 | 2731.0 | 2459.7 | 3.2 | 0.00% |

**Notes / limitations**
- Feature computation is O(history length) per request (full replay of the ledger, capped at
  20,000 rows), so latency grows with ledger size; an incremental state store would make it O(1).
- Scoring is serialized by a lock so history is committed in order; concurrent requests therefore
  queue, which is reflected in the concurrent latencies (throughput does not scale with threads).
