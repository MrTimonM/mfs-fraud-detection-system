"""Local end-to-end prepared-frame inference, explicitly excludes state retrieval."""
import time
import numpy as np
import pandas as pd


def benchmark(bundle, frame, repeats=20):
    records = []
    for batch in sorted({min(n, len(frame)) for n in (1, 10, 100, 1000)}):
        sample = frame.iloc[:min(batch, len(frame))]
        bundle.predict(sample)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            bundle.predict(sample)
            times.append((time.perf_counter()-start)*1000)
        records.append({'batch_size': len(sample), 'mean_ms': np.mean(times), 'median_ms': np.median(times),
            'p50_ms': np.percentile(times, 50), 'p95_ms': np.percentile(times, 95), 'p99_ms': np.percentile(times, 99),
            'transactions_per_second': 1000*len(sample)/np.mean(times), 'repeats': repeats,
            'scope': 'single-process local prepared-frame preprocessing + models + calibration; excludes historical state lookup'})
    return pd.DataFrame(records)
