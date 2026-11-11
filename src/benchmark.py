"""Data Ingestion & Analytical Query Performance Benchmark.

Compares throughput and latency across Pandas, DuckDB, and Polars
for subscription telemetry ingestion, filtering, and cohort aggregations.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import duckdb
import numpy as np
import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    POLARS_AVAILABLE = False

from src.data_loader import generate_synthetic_subscription_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def time_execution(fn: Callable[[], Any], runs: int = 3) -> Dict[str, float]:
    """Measures execution runtime across multiple iterations."""
    durations: List[float] = []
    # Warmup
    _ = fn()
    for _ in range(runs):
        start = time.perf_counter()
        _ = fn()
        durations.append(time.perf_counter() - start)
    return {
        "mean_ms": float(np.mean(durations) * 1000.0),
        "std_ms": float(np.std(durations) * 1000.0),
        "min_ms": float(np.min(durations) * 1000.0),
    }


def benchmark_ingestion(csv_path: Path, runs: int = 3) -> Dict[str, Dict[str, float]]:
    """Benchmarks CSV parsing and loading into memory."""
    results: Dict[str, Dict[str, float]] = {}

    # Pandas
    results["Pandas"] = time_execution(
        lambda: pd.read_csv(csv_path, parse_dates=["signup_date"]), runs=runs
    )

    # DuckDB (Query directly from CSV file without prior loading)
    results["DuckDB (Zero-Copy SQL)"] = time_execution(
        lambda: duckdb.query(f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}')").df(),
        runs=runs,
    )

    # Polars
    if POLARS_AVAILABLE:
        results["Polars"] = time_execution(
            lambda: pl.read_csv(str(csv_path), try_parse_dates=True), runs=runs
        )

    return results


def benchmark_cohort_aggregation(
    csv_path: Path, df_pandas: pd.DataFrame, runs: int = 3
) -> Dict[str, Dict[str, float]]:
    """Benchmarks cohort aggregation: GROUP BY contract_type, internet_service with aggregates."""
    results: Dict[str, Dict[str, float]] = {}

    # 1. Pandas
    def run_pandas():
        return (
            df_pandas.groupby(["contract_type", "internet_service"], as_index=False)
            .agg(
                total_customers=("customer_id", "count"),
                churn_rate=("churn", "mean"),
                avg_monthly_bill=("monthly_charges", "mean"),
                avg_tenure=("tenure_months", "mean"),
            )
            .sort_values("churn_rate", ascending=False)
        )

    results["Pandas"] = time_execution(run_pandas, runs=runs)

    # 2. DuckDB (Direct SQL on In-Memory DataFrame)
    query = """
    SELECT 
        contract_type,
        internet_service,
        COUNT(customer_id) AS total_customers,
        AVG(churn) AS churn_rate,
        AVG(monthly_charges) AS avg_monthly_bill,
        AVG(tenure_months) AS avg_tenure
    FROM df_pandas
    GROUP BY contract_type, internet_service
    ORDER BY churn_rate DESC;
    """
    results["DuckDB (In-Memory SQL)"] = time_execution(
        lambda: duckdb.query(query).df(), runs=runs
    )

    # 3. Polars
    if POLARS_AVAILABLE:
        pl_df = pl.from_pandas(df_pandas)

        def run_polars():
            return (
                pl_df.group_by(["contract_type", "internet_service"])
                .agg(
                    pl.count("customer_id").alias("total_customers"),
                    pl.mean("churn").alias("churn_rate"),
                    pl.mean("monthly_charges").alias("avg_monthly_bill"),
                    pl.mean("tenure_months").alias("avg_tenure"),
                )
                .sort("churn_rate", descending=True)
            )

        results["Polars"] = time_execution(run_polars, runs=runs)

    return results


def run_benchmark_suite(
    num_samples: int = 25000, runs: int = 3, output_markdown: bool = True
) -> str:
    """Executes the complete benchmark suite and produces a formatted markdown report."""
    temp_dir = Path("data") / "processed"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_csv = temp_dir / f"benchmark_data_{num_samples}.csv"

    logger.info("Generating %d benchmark records at %s...", num_samples, temp_csv)
    df_pandas = generate_synthetic_subscription_data(
        num_samples=num_samples, random_state=42, output_path=temp_csv
    )

    logger.info("Starting Ingestion Benchmarks (%d runs)...", runs)
    ingestion_results = benchmark_ingestion(temp_csv, runs=runs)

    logger.info("Starting Cohort Aggregation Benchmarks (%d runs)...", runs)
    agg_results = benchmark_cohort_aggregation(temp_csv, df_pandas, runs=runs)

    # Build report
    pandas_ingest = ingestion_results["Pandas"]["mean_ms"]
    pandas_agg = agg_results["Pandas"]["mean_ms"]

    lines = [
        f"# Ingestion & Analytics Benchmark Report (N = {num_samples:,} records)",
        "",
        "### 1. Raw CSV Ingestion & Parsing",
        "| Engine | Mean Latency (ms) | Min (ms) | Std (ms) | Relative Speedup |",
        "|---|---|---|---|---|",
    ]

    for engine, stats in ingestion_results.items():
        speedup = f"{pandas_ingest / stats['mean_ms']:.2f}x"
        lines.append(
            f"| **{engine}** | {stats['mean_ms']:.2f} ms | {stats['min_ms']:.2f} ms | ±{stats['std_ms']:.2f} ms | {speedup} |"
        )

    lines.extend(
        [
            "",
            "### 2. Multi-Column Cohort Aggregation (Group-By + Rollup)",
            "| Engine | Mean Latency (ms) | Min (ms) | Std (ms) | Relative Speedup |",
            "|---|---|---|---|---|",
        ]
    )

    for engine, stats in agg_results.items():
        speedup = f"{pandas_agg / stats['mean_ms']:.2f}x"
        lines.append(
            f"| **{engine}** | {stats['mean_ms']:.2f} ms | {stats['min_ms']:.2f} ms | ±{stats['std_ms']:.2f} ms | {speedup} |"
        )

    report_content = "\n".join(lines)

    if output_markdown:
        out_file = temp_dir / "benchmark_report.md"
        out_file.write_text(report_content, encoding="utf-8")
        logger.info("Saved benchmark report to %s", out_file.resolve())

    return report_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ingestion & Analytical Query Benchmarks")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")
    args = parser.parse_args()

    report = run_benchmark_suite(num_samples=args.samples, runs=args.runs)
    print("\n" + report)
