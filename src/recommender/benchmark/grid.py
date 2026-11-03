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
from .timing import TimingSettings, block_runner, count_operations, sort_copy, time_call, time_search

# access pattern key -> (hit ratio, position)
ACCESS_PATTERNS = {
    "hits_uniform": (1.0, "uniform"),
    "hits_front": (1.0, "front"),
    "half_misses": (0.5, "uniform"),
    "misses_only": (0.0, "uniform"),
}
SORTED_SEARCHES = ["binary", "binary_branchless", "jump", "exponential", "fibonacci", "interpolation"]
PREPROCESSING = "prep"  # access value of the per-dataset components (sorting, hashing, check)
SAMPLING_PROBES = 10  # probe queries per candidate in the meta-algorithm [vansandt2019]
SAMPLING_TRIALS = 5
QUERY_POOL = 2_000
QUERY_SEED_STEP = 1_000_003
GridConfiguration = DatasetConfig | tuple[DatasetConfig, int]


def _features_row(dataset_id: int, config: DatasetConfig, seed_replicate: int, data, settings: TimingSettings) -> dict:
    features = extract_features(data, seed=0)
    times = [extract_features(data, seed=r).analysis_time_ns for r in range(settings.warmup + settings.repetitions)]
    values = linked_list_to_array(data) if isinstance(data, LinkedList) else np.asarray(data, dtype=np.int64)
    distinct_count = int(np.unique(values).size)
    row = {"dataset_id": dataset_id, **asdict(config)}
    row["seed_replicate"] = seed_replicate
    row.update(
        true_sorted=verify_sorted(data).is_sorted,
        distinct_count=distinct_count,
        duplicate_count=int(values.size - distinct_count),
        actual_duplicate_share=float((values.size - distinct_count) / values.size) if values.size else 0.0,
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


def _sampling_choice(candidates: dict, data, probes: np.ndarray) -> tuple[str, float]:
    """Meta-algorithm of [vansandt2019]: time every candidate on a few probe queries, pick the fastest."""
    ops = new_counter()
    total = 0.0
    best_key, best_ns = "", float("inf")
    for key, algorithm in candidates.items():
        runner = block_runner(algorithm)
        start = time.perf_counter_ns()
        runner(data, probes, ops)
        elapsed = time.perf_counter_ns() - start
        total += elapsed
        if elapsed < best_ns:
            best_key, best_ns = key, elapsed
    return best_key, total


def measure_dataset(
    dataset_id: int,
    config: DatasetConfig,
    settings: TimingSettings,
    seed_replicate: int = 0,
):
    """Measure one dataset; return features, time components, operations, and sampling rows."""
    data = generate_dataset(config)
    features = _features_row(dataset_id, config, seed_replicate, data, settings)
    is_sorted = features["true_sorted"]
    components = []
    operations = []

    def add(
        access: str,
        query_id: str,
        query_seed: int,
        hit_ratio: float,
        position: str,
        component: str,
        samples,
        block_queries: int | None = None,
    ) -> None:
        components.extend(
            {
                "dataset_id": dataset_id,
                "seed_replicate": seed_replicate,
                "access": access,
                "query_id": query_id,
                "query_seed": query_seed,
                "query_pool_size": QUERY_POOL if access != PREPROCESSING else np.nan,
                "hit_ratio": hit_ratio,
                "position": position,
                "component": component,
                "repetition": repetition,
                "block_queries": block_queries,
                "ns": float(ns),
            }
            for repetition, ns in enumerate(samples)
        )

    add(PREPROCESSING, "", config.seed, np.nan, "", "verify", _verify_samples(data, settings))
    if isinstance(data, LinkedList):
        table = build_hash_table(linked_list_to_array(data))
        add(
            PREPROCESSING,
            "",
            config.seed,
            np.nan,
            "",
            "list_hash_build",
            time_call(lambda: build_hash_table(linked_list_to_array(data)), settings),
        )
        searches = {"list_linear": (LIST_ALGORITHMS["list_linear"], data.as_tuple())}
        if is_sorted:
            searches["list_jump"] = (LIST_ALGORITHMS["list_jump"], data.as_tuple())
        searches["hash"] = (ALGORITHMS["hash"], table)
    else:
        table = build_hash_table(data)
        sorted_data = data if is_sorted else sort_copy(data)
        add(PREPROCESSING, "", config.seed, np.nan, "", "sort", time_call(lambda: sort_copy(data), settings))
        add(
            PREPROCESSING,
            "",
            config.seed,
            np.nan,
            "",
            "hash_build",
            time_call(lambda: build_hash_table(data), settings),
        )
        searches = {"linear": (ALGORITHMS["linear"], data), "hash": (ALGORITHMS["hash"], table)}
        if is_sorted:
            searches.update({key: (ALGORITHMS[key], data) for key in SORTED_SEARCHES})
        searches["after_sort_binary_branchless"] = (ALGORITHMS["binary_branchless"], sorted_data)

    sampling = []
    for access_index, (access, (hit_ratio, position)) in enumerate(ACCESS_PATTERNS.items()):
        query_seed = config.seed + access_index * QUERY_SEED_STEP
        query_id = f"{dataset_id}:{access}"
        pool = generate_queries(
            data,
            QueryConfig(count=QUERY_POOL, hit_ratio=hit_ratio, position=position, seed=query_seed),
        )
        block_rng = np.random.default_rng(query_seed)
        query_blocks = [
            block_rng.choice(pool, settings.max_block_queries)
            for _ in range(settings.warmup + settings.repetitions)
        ]
        for algorithm_key, (algorithm, prepared) in searches.items():
            samples, block_length = time_search(
                algorithm,
                prepared,
                pool,
                settings,
                block_rng,
                query_blocks=query_blocks,
            )
            add(
                access,
                query_id,
                query_seed,
                hit_ratio,
                position,
                f"q_{algorithm_key}",
                samples,
                block_queries=block_length,
            )
            comparisons, accesses = count_operations(algorithm, prepared, pool, settings)
            operations.append(
                {
                    "dataset_id": dataset_id,
                    "seed_replicate": seed_replicate,
                    "access": access,
                    "query_id": query_id,
                    "query_seed": query_seed,
                    "query_count": min(settings.count_queries, pool.size),
                    "hit_ratio": hit_ratio,
                    "position": position,
                    "algorithm": algorithm_key,
                    "comparisons": comparisons,
                    "accesses": accesses,
                }
            )
        if is_sorted and not isinstance(data, LinkedList):
            candidates = {key: ALGORITHMS[key] for key in ["linear", *SORTED_SEARCHES]}
            for trial in range(SAMPLING_TRIALS):
                probe_rng = np.random.default_rng(np.random.SeedSequence([query_seed, trial]))
                probes = probe_rng.choice(pool, SAMPLING_PROBES)
                chosen, probe_ns = _sampling_choice(candidates, data, probes)
                sampling.append(
                    {
                        "dataset_id": dataset_id,
                        "seed_replicate": seed_replicate,
                        "access": access,
                        "query_id": query_id,
                        "query_seed": query_seed,
                        "trial": trial,
                        "chosen": chosen,
                        "probe_ns": probe_ns,
                    }
                )
    return features, components, operations, sampling


def measure_grid(
    configs: list[GridConfiguration],
    settings: TimingSettings,
    seed: int,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure datasets in random order; return features, components, operations, sampling."""
    rng = np.random.default_rng(seed)
    features, components, operations, sampling = [], [], [], []
    order = rng.permutation(len(configs))
    for step, index in enumerate(order, start=1):
        item = configs[index]
        if isinstance(item, DatasetConfig):
            config, replicate = item, 0
        else:
            config, replicate = item
        if progress:
            print(f"[{step}/{len(configs)}] {config} (seed replicate {replicate})", flush=True)
        f, c, o, s = measure_dataset(int(index), config, settings, seed_replicate=replicate)
        features.append(f)
        components.extend(c)
        operations.extend(o)
        sampling.extend(s)
    return pd.DataFrame(features), pd.DataFrame(components), pd.DataFrame(operations), pd.DataFrame(sampling)
