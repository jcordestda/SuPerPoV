#!/usr/bin/env python3
"""Profile the original CPU calculation without importing its GUI entry point.

Example:
    python benchmarks/profile_cpu.py \
        --file gph_winter_1965.txt --dates 1965-12-23 --repeats 3 \
        --output /tmp/superpov-profile.json

Only the three calculation functions are compiled from the original source.
The instrumented variant wraps unchanged, contiguous statements in timers.
Optional cProfile runs are additional runs, excluded from normal timings.
"""

import argparse
import ast
import copy
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

from input_validation import validate_frame, validate_output_path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "SuPerPoV3.1.0.py"
FUNCTION_NAMES = ("correct_df", "date_index_returner", "compute_data")
# These ranges deliberately describe this source revision. Refuse unmapped or
# straddling statements instead of silently attributing changed code incorrectly.
STAGES = (
    ("date_routing", 207, 214),
    ("read_winter", 217, 217),
    ("day_lookup", 220, 220),
    ("extract_day", 222, 224),
    ("polar_geometry", 227, 237),
    ("polar_simplex_construction_and_assignment", 239, 250),
    ("polar_make_filtration", 251, 251),
    ("polar_extend_filtration", 252, 252),
    ("polar_persistence", 253, 253),
    ("polar_diagram_processing", 256, 261),
    ("duplicate_grid_values", 264, 265),
    ("grid_geometry", 267, 273),
    ("grid_simplex_construction_and_assignment", 275, 286),
    ("grid_make_filtration", 287, 287),
    ("grid_extend_filtration", 288, 288),
    ("grid_persistence", 289, 289),
    ("grid_diagram_processing_and_scores", 292, 318),
)


class StageTimer:
    def __init__(self, samples, name):
        self.samples = samples
        self.name = name

    def __enter__(self):
        self.started = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, traceback):
        self.samples[self.name] = (time.perf_counter_ns() - self.started) / 1e9
        return False


def compile_functions(source_text, dependencies, instrumented=False):
    """Compile only selected definitions; never execute application imports/main."""
    parsed = ast.parse(source_text, filename=str(SOURCE_PATH))
    functions = [
        copy.deepcopy(node)
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
    ]
    if tuple(node.name for node in functions) != FUNCTION_NAMES:
        raise ValueError("Expected calculation function definitions were not found")
    if instrumented:
        compute = functions[-1]
        original_statements = list(compute.body)
        original_dump = [ast.dump(node) for node in original_statements]
        wrapped = []
        assigned = []
        for name, first_line, last_line in STAGES:
            statements = [
                node for node in original_statements
                if first_line <= node.lineno <= last_line
            ]
            if not statements or any(node.end_lineno > last_line for node in statements):
                raise ValueError(f"Source no longer matches stage {name!r}")
            assigned.extend(statements)
            context = ast.Call(
                func=ast.Name(id="_stage_timer", ctx=ast.Load()),
                args=[ast.Constant(value=name)],
                keywords=[],
            )
            wrapper = ast.With(
                items=[ast.withitem(context_expr=context, optional_vars=None)],
                body=statements,
                type_comment=None,
            )
            ast.copy_location(wrapper, statements[0])
            wrapper.end_lineno = statements[-1].end_lineno
            wrapper.end_col_offset = statements[-1].end_col_offset
            wrapped.append(wrapper)
        if [ast.dump(node) for node in assigned] != original_dump:
            raise ValueError("Instrumentation must preserve every statement in order")
        compute.body = wrapped
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = dict(dependencies)
    exec(compile(module, filename=str(SOURCE_PATH), mode="exec"), namespace)
    return namespace


def warm_file_and_hash(path):
    """Read the whole input before measurements; do not claim a cold-cache test."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def summary(values):
    return {
        "count": len(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path, help="One winter TSV file")
    parser.add_argument("--dates", nargs="+", default=["1965-12-23"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resolution", nargs=2, type=int, default=[46, 180],
                        metavar=("LATITUDES", "LONGITUDES"))
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument("--cprofile-dir", type=Path,
                        help="Optional directory for separate .prof and text profiles")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.resolution) < 2:
        parser.error("Use at least one repetition and a resolution of at least 2 x 2")
    for date in args.dates:
        try:
            parsed = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            parser.error(f"Invalid date: {date}")
        if parsed.strftime("%Y-%m-%d") != date:
            parser.error(f"Date must use YYYY-MM-DD: {date}")
    if len(set(args.dates)) != len(args.dates):
        parser.error("Dates must be unique")
    if not args.file.is_file():
        parser.error(f"Input file does not exist: {args.file}")
    args.file = args.file.resolve()
    try:
        protected = (args.file, SOURCE_PATH, Path(__file__))
        args.output = validate_output_path(args.output, protected)
        if args.cprofile_dir:
            for date in args.dates:
                for suffix in ('.prof', '.txt'):
                    validate_output_path(args.cprofile_dir / f'cpu-{date}{suffix}',
                                         (*protected, args.output))
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    import_started = time.perf_counter()
    import gudhi as gd
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.spatial import Delaunay
    dependency_import_seconds = time.perf_counter() - import_started

    validation_started = time.perf_counter()
    try:
        preflight_frame = pd.read_csv(args.file, sep='\t')
        input_validation = validate_frame(preflight_frame, tuple(args.resolution),
                                          args.dates, original_lookup=True)
    except (ValueError, OSError) as error:
        raise SystemExit(f'Invalid input: {error}') from error
    del preflight_frame
    validation_seconds = time.perf_counter() - validation_started

    source_text = SOURCE_PATH.read_text()
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    dependencies = {"pd": pd, "np": np, "gd": gd, "Delaunay": Delaunay}
    original = compile_functions(source_text, dependencies)["compute_data"]
    samples = {}
    instrumented_namespace = compile_functions(source_text, dependencies, instrumented=True)
    instrumented_namespace["_stage_timer"] = lambda name: StageTimer(samples, name)
    instrumented = instrumented_namespace["compute_data"]
    input_hash = warm_file_and_hash(args.file)
    txt_files = [str(args.file)]
    resolution = tuple(args.resolution)
    report = {
        "metadata": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "pandas": pd.__version__, "gudhi": gd.__version__},
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "source": str(SOURCE_PATH),
            "source_sha256": source_hash,
            "input_file": str(args.file),
            "input_sha256": input_hash,
            "input_bytes": args.file.stat().st_size,
            "resolution": args.resolution,
            "dates": args.dates,
            "repeats": args.repeats,
            "dependency_import_seconds_excluded": dependency_import_seconds,
            "input_validation_seconds_excluded": validation_seconds,
            "input_validation": input_validation,
            "timer": "time.perf_counter_ns; seconds reported",
        },
        "methodology": {
            "scope": "Original compute_data and its two original data helpers",
            "excluded": ["GUI", "plotting", "wind data", "application main loop",
                         "imports", "input validation", "AST extraction/compilation", "input hashing/cache warming",
                         "output serialization", "optional cProfile runs"],
            "filesystem_cache": "Preflight parsing and full-file hashing before timed calls; warm-cache measurements, no cache flush",
            "computation_warmup": "No topology warmup; preflight warms the parser and filesystem cache; first topology-call effects are retained",
            "file_search_scope": "One explicitly supplied winter file; original matching-file double parse retained",
            "order": "Alternate original-first and instrumented-first on successive repetitions",
            "equality": "Exact Python equality of complete returned lists; no tolerance",
            "instrumentation": "Unmodified original statements grouped in contiguous timing contexts",
            "stage_timing_note": "Stage totals exclude some context/dispatch overhead and function cleanup; use uninstrumented elapsed for baseline",
            "cprofile": "One additional uninstrumented-function call per date when requested, reported separately",
        },
        "stages": [{"name": name, "source_first_line": first, "source_last_line": last}
                   for name, first, last in STAGES],
        "runs": [],
        "cprofile_runs": [],
    }

    for date in args.dates:
        for repeat in range(1, args.repeats + 1):
            variants = ("original", "instrumented") if repeat % 2 else ("instrumented", "original")
            run = {"date": date, "repeat": repeat, "execution_order": list(variants)}
            outputs = {}
            for variant in variants:
                samples.clear()
                function = original if variant == "original" else instrumented
                started = time.perf_counter_ns()
                outputs[variant] = function(date, txt_files, resolution)
                elapsed = (time.perf_counter_ns() - started) / 1e9
                run[f"{variant}_elapsed_seconds"] = elapsed
                if variant == "instrumented":
                    if list(samples) != [stage[0] for stage in STAGES]:
                        raise RuntimeError("Missing or misordered instrumented stage timings")
                    run["stage_seconds"] = dict(samples)
                    run["stage_sum_seconds"] = sum(samples.values())
                    run["elapsed_outside_stages_seconds"] = elapsed - sum(samples.values())
            run["outputs_equal_exactly"] = outputs["original"] == outputs["instrumented"]
            run["original_output"] = json_value(outputs["original"])
            run["instrumented_output"] = json_value(outputs["instrumented"])
            report["runs"].append(run)
            print(f"{date} repeat {repeat}: original={run['original_elapsed_seconds']:.6f}s "
                  f"instrumented={run['instrumented_elapsed_seconds']:.6f}s "
                  f"equal={run['outputs_equal_exactly']}", flush=True)

    if args.cprofile_dir:
        args.cprofile_dir.mkdir(parents=True, exist_ok=True)
        for date in args.dates:
            profiler = cProfile.Profile()
            started = time.perf_counter_ns()
            output = profiler.runcall(original, date, txt_files, resolution)
            elapsed = (time.perf_counter_ns() - started) / 1e9
            profile_path = (args.cprofile_dir / f"cpu-{date}.prof").resolve()
            text_path = profile_path.with_suffix(".txt")
            profiler.dump_stats(str(profile_path))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(40)
            stats.sort_stats("tottime").print_stats(40)
            text_path.write_text(stream.getvalue())
            reference = next(run["original_output"] for run in report["runs"] if run["date"] == date)
            report["cprofile_runs"].append({
                "date": date,
                "elapsed_seconds_with_profiler_overhead": elapsed,
                "output": json_value(output),
                "output_equals_reference_exactly": json_value(output) == reference,
                "profile_path": str(profile_path),
                "text_path": str(text_path),
            })

    report["summary_by_date"] = {}
    for date in args.dates:
        runs = [run for run in report["runs"] if run["date"] == date]
        report["summary_by_date"][date] = {
            "original": summary([run["original_elapsed_seconds"] for run in runs]),
            "instrumented": summary([run["instrumented_elapsed_seconds"] for run in runs]),
            "stages": {name: summary([run["stage_seconds"][name] for run in runs])
                       for name, _, _ in STAGES},
            "all_original_repetitions_equal_exactly": all(
                run["original_output"] == runs[0]["original_output"] for run in runs),
        }
    report["all_outputs_equal_exactly"] = (
        all(run["outputs_equal_exactly"] for run in report["runs"])
        and all(run["output_equals_reference_exactly"] for run in report["cprofile_runs"])
        and all(item["all_original_repetitions_equal_exactly"]
                for item in report["summary_by_date"].values())
    )
    report["metadata"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)
    if not report["all_outputs_equal_exactly"]:
        raise SystemExit("Output equality check failed; see the JSON artifact")


if __name__ == "__main__":
    main()
