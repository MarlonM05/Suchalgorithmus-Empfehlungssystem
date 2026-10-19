"""Measurement grid for training and evaluating the recommender.

For every dataset of the grid, all **cost components** are measured that the
strategies (see rules/strategies.py) are made of:

* preprocessing per dataset: full sortedness check, sorting, building the
  hash table (array) or copying the list into a hash table,
* cost per query of every applicable search algorithm, per access pattern,
* the estimated features of the dataset and the cost of the analysis.

The total cost of a strategy for any number of queries q is then computed with
the linear cost model ``preprocessing + q * cost per query`` (docs/annahmen.md
A22), so q does not have to be measured separately.

Additionally, the sampling meta-algorithm of [vansandt2019] is run on sorted
arrays: every candidate algorithm answers a few probe queries, the fastest
one is chosen.
"""

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..algorithms import ALGORITHMS, LIST_ALGORITHMS, new_counter
from ..algorithms.hashing import build_hash_table
from ..data import DatasetConfig, LinkedList, QueryConfig, generate_dataset, generate_queries, linked_list_to_array
from ..features import extract_features, verify_sorted
from .timing import TimingSettings, block_runner, sort_copy, time_call, time_search

# access pattern key -> (hit ratio, position)
ACCESS_PATTERNS = {
    "hits_uniform": (1.0, "uniform"),
    "hits_front": (1.0, "front"),
    "half_misses": (0.5, "uniform"),
}
SORTED_SEARCHES = ["binary", "binary_branchless", "jump", "exponential", "fibonacci", "interpolation"]
PREPROCESSING = "prep"  # access value of the per-dataset components (sorting, hashing, check)
SAMPLING_PROBES = 10  # probe queries per candidate in the meta-algorithm [vansandt2019]
SAMPLING_TRIALS = 5
QUERY_POOL = 2_000


def _features_row(dataset_id: int, config: DatasetConfig, data, settings: TimingSettings) -> dict:
    features = extract_features(data, seed=0)
    times = [extract_features(data, seed=r).analysis_time_ns for r in range(settings.warmup + settings.repetitions)]
    row = {"dataset_id": dataset_id, **asdict(config)}
    row.update(
        true_sorted=verify_sorted(data).is_sorted,
        est_n=features.n,
        disorder=features.disorder,
        ks_uniform=features.ks_uniform,
        duplicate_share_est=-1.0 if features.duplicate_share is None else features.duplicate_share,
        sample_size=features.sample_size,
        analysis_ns=float(np.median(times[settings.warmup :])),
    )
    return row


def _verify_samples(data, settings: TimingSettings) -> np.ndarray:
    runs = [verify_sorted(data).time_ns for _ in range(settings.warmup + settings.repetitions)]
    return np.array(runs[settings.warmup :], dtype=np.float64)


def _sampling_choice(candidates: dict, data, pool: np.ndarray, rng: np.random.Generator) -> tuple[str, float]:
    """Meta-algorithm of [vansandt2019]: time every candidate on a few probe queries, pick the fastest."""
    ops = new_counter()
    total = 0.0
    best_key, best_ns = "", float("inf")
    for key, algorithm in candidates.items():
        runner = block_runner(algorithm)
        probes = rng.choice(pool, SAMPLING_PROBES)
        start = time.perf_counter_ns()
        runner(data, probes, ops)
        elapsed = time.perf_counter_ns() - start
        total += elapsed
        if elapsed < best_ns:
            best_key, best_ns = key, elapsed
    return best_key, total


def measure_dataset(dataset_id: int, config: DatasetConfig, settings: TimingSettings, rng: np.random.Generator):
    """Measure all components of one dataset; returns (features row, component rows, sampling rows)."""
    data = generate_dataset(config)
    features = _features_row(dataset_id, config, data, settings)
    is_sorted = features["true_sorted"]
    components = []

    def add(access: str, component: str, samples) -> None:
        components.extend(
            {"dataset_id": dataset_id, "access": access, "component": component, "repetition": r, "ns": float(ns)}
            for r, ns in enumerate(samples)
        )

    add(PREPROCESSING, "verify", _verify_samples(data, settings))
    if isinstance(data, LinkedList):
        table = build_hash_table(linked_list_to_array(data))
        add(PREPROCESSING, "list_hash_build", time_call(lambda: build_hash_table(linked_list_to_array(data)), settings))
        searches = {"list_linear": (LIST_ALGORITHMS["list_linear"], data.as_tuple())}
        if is_sorted:
            searches["list_jump"] = (LIST_ALGORITHMS["list_jump"], data.as_tuple())
        searches["hash"] = (ALGORITHMS["hash"], table)
    else:
        table = build_hash_table(data)
        sorted_data = data if is_sorted else sort_copy(data)
        add(PREPROCESSING, "sort", time_call(lambda: sort_copy(data), settings))
        add(PREPROCESSING, "hash_build", time_call(lambda: build_hash_table(data), settings))
        searches = {"linear": (ALGORITHMS["linear"], data), "hash": (ALGORITHMS["hash"], table)}
        if is_sorted:
            searches.update({key: (ALGORITHMS[key], data) for key in SORTED_SEARCHES})
        searches["after_sort_binary_branchless"] = (ALGORITHMS["binary_branchless"], sorted_data)

    sampling = []
    for access, (hit_ratio, position) in ACCESS_PATTERNS.items():
        pool = generate_queries(data, QueryConfig(count=QUERY_POOL, hit_ratio=hit_ratio, position=position, seed=dataset_id))
        for component, (algorithm, prepared) in searches.items():
            add(access, f"q_{component}", time_search(algorithm, prepared, pool, settings, rng)[0])
        if is_sorted and not isinstance(data, LinkedList):
            candidates = {key: ALGORITHMS[key] for key in ["linear", *SORTED_SEARCHES]}
            for trial in range(SAMPLING_TRIALS):
                chosen, probe_ns = _sampling_choice(candidates, data, pool, rng)
                sampling.append({"dataset_id": dataset_id, "access": access, "trial": trial, "chosen": chosen, "probe_ns": probe_ns})
    return features, components, sampling


def measure_grid(configs: list[DatasetConfig], settings: TimingSettings, seed: int, progress: bool = True):
    """Measure all datasets in random order; returns three tables (features, components, sampling)."""
    rng = np.random.default_rng(seed)
    features, components, sampling = [], [], []
    order = rng.permutation(len(configs))
    for step, index in enumerate(order, start=1):
        if progress:
            print(f"[{step}/{len(configs)}] {configs[index]}", flush=True)
        f, c, s = measure_dataset(int(index), configs[index], settings, rng)
        features.append(f)
        components.extend(c)
        sampling.extend(s)
    return pd.DataFrame(features), pd.DataFrame(components), pd.DataFrame(sampling)
