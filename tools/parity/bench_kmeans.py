#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 TPT Solutions
"""Times scikit-learn KMeans on the same data as the Rust benchmark.

    python tools/parity/bench_kmeans.py

Writes the data file (once) under target/bench/, which is git-ignored, then times
KMeans.fit on it. Both sides use the same shape, the same starting centres (the
first k rows), tol=0 and max_iter, so they do the same work. The reported time is
the best of three runs. Thread counts are 1 and all cores.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROWS, COLS, K, ITERS = 1_000_000, 50, 8, 20
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "target" / "bench" / f"kmeans_{ROWS}x{COLS}_f32.bin"


def load_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        rng.standard_normal((ROWS, COLS), dtype=np.float32).astype("<f4").tofile(DATA)
    return np.fromfile(DATA, dtype="<f4").reshape(ROWS, COLS)


def time_fit(x, threads):
    init = x[:K].copy()
    best = float("inf")
    n_iter = None
    for _ in range(3):
        model = KMeans(
            n_clusters=K,
            init=init,
            n_init=1,
            algorithm="lloyd",
            tol=0.0,
            max_iter=ITERS,
        )
        with threadpool_limits(limits=threads):
            start = time.perf_counter()
            model.fit(x)
            best = min(best, time.perf_counter() - start)
        n_iter = int(model.n_iter_)
    return best, n_iter


def main():
    x = load_data()
    cores = os.cpu_count() or 1
    results = []
    for threads in sorted({1, cores}):
        seconds, n_iter = time_fit(x, threads)
        results.append(
            {
                "impl": f"scikit-learn {sklearn.__version__}",
                "threads": threads,
                "seconds": round(seconds, 4),
                "n_iter": n_iter,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
