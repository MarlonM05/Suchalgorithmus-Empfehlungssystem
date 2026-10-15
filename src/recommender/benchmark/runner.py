"""Run a series of measurements and collect all raw values in a table.

A *configuration* is a dataset plus an access pattern. For every
configuration, all applicable search algorithms are measured: algorithms that
require sorted data only on sorted data, list algorithms only on linked lists.

To avoid measurement bias, the order of the configurations and the order of
the algorithms within a configuration are randomised with a fixed, logged
seed [mytkowicz2009].
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..algorithms import ALGORITHMS, LIST_ALGORITHMS, SearchAlgorithm
from ..data import DatasetConfig, QueryConfig, generate_dataset, generate_queries
from .timing import TimingSettings, count_operations, time_search


@dataclass(frozen=True)
class Configuration:
    """One measured setting: dataset characteristics plus access pattern."""

    dataset: DatasetConfig
    queries: QueryConfig

    def as_columns(self) -> dict:
        """Flat column values for the result table."""
        columns = asdict(self.dataset)
        columns.pop("seed")
        columns["dataset_seed"] = self.dataset.seed
        columns.update(
            query_count=self.queries.count,
            hit_ratio=self.queries.hit_ratio,
            position=self.queries.position,
            query_seed=self.queries.seed,
        )
        return columns


def applicable_algorithms(dataset: DatasetConfig) -> list[SearchAlgorithm]:
    """Algorithms whose preconditions hold for the dataset."""
    if dataset.structure == "linked_list":
        candidates = LIST_ALGORITHMS.values()
    else:
        candidates = ALGORITHMS.values()
    is_sorted = dataset.order == "sorted"
    return [a for a in candidates if is_sorted or not a.requires_sorted]


def run_configurations(
    configurations: list[Configuration],
    settings: TimingSettings,
    seed: int = 0,
    algorithms: list[str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Measure all applicable algorithms on all configurations.

    Args:
        configurations: settings to measure.
        settings: timing parameters.
        seed: seed for the randomised order and the key blocks.
        algorithms: optional subset of algorithm keys to measure.
        progress: print one line per configuration.

    Returns:
        One row per configuration, algorithm and repetition with the raw
        nanoseconds per query plus the mean operation counts.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(configurations))
    rows = []
    for step, index in enumerate(order, start=1):
        configuration = configurations[index]
        data = generate_dataset(configuration.dataset)
        query_pool = generate_queries(data, configuration.queries)
        candidates = [a for a in applicable_algorithms(configuration.dataset) if algorithms is None or a.key in algorithms]
        if progress:
            print(f"[{step}/{len(configurations)}] {configuration.as_columns()}", flush=True)

        for algorithm_index in rng.permutation(len(candidates)):
            algorithm = candidates[algorithm_index]
            prepared = algorithm.prepare(data)
            samples, block_length = time_search(algorithm, prepared, query_pool, settings, rng)
            comparisons, accesses = count_operations(algorithm, prepared, query_pool, settings)
            for repetition, ns in enumerate(samples):
                rows.append(
                    {
                        **configuration.as_columns(),
                        "algorithm": algorithm.key,
                        "repetition": repetition,
                        "ns_per_query": ns,
                        "block_queries": block_length,
                        "comparisons": comparisons,
                        "accesses": accesses,
                    }
                )
    return pd.DataFrame(rows)
