#!/usr/bin/env python3
"""Compare the original CPU calculation, engine methods, cache, and real CLI.

    python superpov_planar.py --build
    python benchmarks/compare_cpu.py --file gph_winter_1965.txt --output results.json

All trials are sequential and use warm filesystem caches. Build the native
planar backend before running: compilation is excluded from these timings.
Persistent-cache creation is measured separately in a temporary directory.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time

import compare_daily
from input_validation import validate_frame, validate_output_path
import profile_cpu


ROOT = Path(__file__).resolve().parents[1]
COMPUTE_PATH = ROOT / 'superpov_compute.py'
RESOLUTION = compare_daily.RESOLUTION
DEFAULT_DATES = compare_daily.DEFAULT_DATES
persistence = compare_daily.persistence
load_analyzer_class = compare_daily.load_analyzer_class
CASES = ('original', 'direct_fresh', 'planar_fresh', 'planar_disk_hit',
         'direct_warm_different_day', 'planar_warm_different_day', 'planar_cli_disk_hit')


def source_paths():
    return (profile_cpu.SOURCE_PATH, COMPUTE_PATH, ROOT / 'superpov_cache.py',
            ROOT / 'superpov_planar.py', ROOT / 'superpov_planar.cpp', Path(__file__),
            Path(compare_daily.__file__), Path(profile_cpu.__file__),
            Path(persistence.__file__), ROOT / 'benchmarks' / 'fast_build.py',
            ROOT / 'benchmarks' / 'input_validation.py')


def source_hashes():
    return {path.name: profile_cpu.warm_file_and_hash(path) for path in source_paths()}


def cache_bytes(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())


def native_build_metadata():
    """Report the available compiler; never compile inside timing trials."""
    command = []
    try:
        command = shlex.split(os.environ.get('CXX', 'c++'))
        result = subprocess.run(command + ['--version'], capture_output=True,
                                text=True, check=True)
        version = result.stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        version = None
    return {
        'build_command': 'python superpov_planar.py --build',
        'build_flags': ['-std=c++11', '-O3', '-DNDEBUG', '-fPIC',
                        '-dynamiclib' if sys.platform == 'darwin' else '-shared'],
        'scope': 'Native backend must be explicitly prebuilt; compilation is excluded from all timings',
        'compiler_available_at_benchmark': Path(command[0]).name if command else None,
        'compiler_version_available_at_benchmark': version,
        'compiler_note': 'Available compiler is recorded for reproducibility; this does not independently attest which compiler produced a pre-existing binary',
    }


def run_cli(date, txt_files, resolution, cache_dir, pressure=None):
    """Launch a complete new Python process and parse its single JSON result."""
    command = [sys.executable, str(COMPUTE_PATH), '--file', *map(str, txt_files),
               '--dates', date, '--resolution', *map(str, resolution),
               '--method', 'planar', '--cache-dir', str(cache_dir)]
    if pressure is not None:
        command += ['--pressure', str(pressure)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError('CLI must produce exactly one JSON result for one date')
    payload = json.loads(lines[0])
    if not isinstance(payload, dict) or not {
            'date', 'depths', 'split', 'displacement', 'elapsed_seconds'} <= payload.keys():
        raise ValueError('CLI result is missing date, scores, or elapsed_seconds')
    return payload


def run_timing_trials(original, analyzer_factory, txt_files, resolution, dates,
                      available_dates, repeats, cache_dir, pressure=None,
                      cli_runner=run_cli):
    """Retain all results and compare each candidate with its own baseline run."""
    if repeats < 1 or not dates:
        raise ValueError('At least one date and repetition are required')
    if len(set(available_dates)) < 2:
        raise ValueError('Warm measurements require a distinct, different seed date')
    if cache_bytes(cache_dir):
        raise ValueError('Persistent-cache creation requires an empty cache directory')

    started = time.perf_counter()
    analyzer = analyzer_factory(resolution=resolution, method='planar', cache_dir=cache_dir)
    prepared = analyzer.compute_data(dates[0], txt_files, pressure=pressure)
    creation_seconds = time.perf_counter() - started
    del analyzer
    creation = {'date': dates[0], 'seconds': creation_seconds,
                'bytes': cache_bytes(cache_dir), 'output': profile_cpu.json_value(prepared)}
    preferred_seeds = [date for date in available_dates if date not in dates]
    first_original = {}
    runs = []
    for date in dates:
        seed = next(day for day in (preferred_seeds or available_dates) if day != date)
        for repeat in range(repeats):
            start = repeat % len(CASES)
            order = CASES[start:] + CASES[:start]
            elapsed, outputs = {}, {}
            cli_payload = None
            for case in order:
                if case in ('direct_warm_different_day', 'planar_warm_different_day'):
                    method = 'direct' if case == 'direct_warm_different_day' else 'planar'
                    analyzer = analyzer_factory(resolution=resolution, method=method, cache_dir=None)
                    analyzer.compute_data(seed, txt_files, pressure=pressure)
                    started = time.perf_counter()
                    output = analyzer.compute_data(date, txt_files, pressure=pressure)
                    elapsed[case] = time.perf_counter() - started
                    del analyzer
                elif case == 'original':
                    started = time.perf_counter()
                    output = original(date, txt_files, resolution)
                    elapsed[case] = time.perf_counter() - started
                elif case == 'planar_cli_disk_hit':
                    started = time.perf_counter()
                    cli_payload = cli_runner(date, txt_files, resolution, cache_dir, pressure=pressure)
                    elapsed[case] = time.perf_counter() - started
                    output = [cli_payload['depths'], cli_payload['split'], cli_payload['displacement']]
                else:
                    method = 'direct' if case == 'direct_fresh' else 'planar'
                    directory = cache_dir if case == 'planar_disk_hit' else None
                    started = time.perf_counter()
                    analyzer = analyzer_factory(resolution=resolution, method=method, cache_dir=directory)
                    output = analyzer.compute_data(date, txt_files, pressure=pressure)
                    elapsed[case] = time.perf_counter() - started
                    del analyzer
                outputs[case] = profile_cpu.json_value(output)
            first_original.setdefault(date, outputs['original'])
            checks = {case + '_exact': outputs[case] == outputs['original']
                      for case in CASES if case != 'original'}
            checks['planar_cli_disk_hit_exact'] &= cli_payload['date'] == date
            checks['original_repeat_exact'] = outputs['original'] == first_original[date]
            checks['passed'] = all(checks.values())
            run = {'date': date, 'repeat': repeat + 1, 'warmup_date': seed,
                   'execution_order': list(order), 'seconds': elapsed, 'outputs': outputs,
                   'cli_payload': cli_payload, 'verification': checks}
            runs.append(run)
            print(json.dumps({'date': date, 'repeat': repeat + 1, 'seconds': elapsed,
                              'verification': checks}), flush=True)
    creation['verification'] = {
        'original_exact': creation['output'] == first_original[creation['date']],
        'cache_written': creation['bytes'] > 0,
    }
    creation['verification']['passed'] = all(creation['verification'].values())
    creation['final_bytes'] = cache_bytes(cache_dir)
    return runs, creation


def summarize_timings(runs):
    summaries = {}
    for date in dict.fromkeys(run['date'] for run in runs):
        selected = [run for run in runs if run['date'] == date]
        cases = {case: profile_cpu.summary([run['seconds'][case] for run in selected])
                 for case in CASES}
        baseline = cases['original']['median_seconds']
        summaries[date] = {
            'cases': cases,
            'speedup_from_original_medians': {
                case: baseline / cases[case]['median_seconds'] for case in CASES if case != 'original'},
            'planar_speedup_from_direct_medians': {
                'fresh': cases['direct_fresh']['median_seconds'] / cases['planar_fresh']['median_seconds'],
                'warm_different_day': cases['direct_warm_different_day']['median_seconds']
                / cases['planar_warm_different_day']['median_seconds']},
        }
    return summaries


def report_passed(runs, creation, input_unchanged, sources_unchanged):
    return (bool(runs) and input_unchanged and sources_unchanged
            and creation['verification']['passed']
            and all(run['verification']['passed'] for run in runs))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True, help='One 46 x 180 single-pressure winter TSV')
    parser.add_argument('--dates', nargs='+', default=list(DEFAULT_DATES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks' / 'results' / 'cpu_comparison.json')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    if not args.file.is_file():
        parser.error(f'Input file does not exist: {args.file}')
    args.file = args.file.resolve()
    try:
        args.output = validate_output_path(args.output, (args.file, *source_paths()))
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        frame = persistence.pd.read_csv(args.file, sep='\t')
        checked = validate_frame(frame, RESOLUTION, args.dates, original_lookup=True)
        if len(checked['dates']) < 2:
            raise ValueError('Warm measurements require a distinct, different seed date')
    except (ValueError, OSError) as error:
        raise SystemExit(f'Invalid input: {error}') from error
    del frame
    input_hash = profile_cpu.warm_file_and_hash(args.file)
    hashes = source_hashes()
    analyzer_class = load_analyzer_class()
    native = native_build_metadata()
    dependencies = {'pd': persistence.pd, 'np': persistence.np,
                    'gd': persistence.gd, 'Delaunay': persistence.Delaunay}
    original = profile_cpu.compile_functions(profile_cpu.SOURCE_PATH.read_text(), dependencies)['compute_data']
    metadata = {
        'started_at_utc': started_utc, 'source_sha256': hashes,
        'input': {'name': args.file.name, 'sha256': input_hash, 'bytes': args.file.stat().st_size,
                  'days': len(checked['dates']), 'date_range': [checked['dates'][0], checked['dates'][-1]],
                  'pressure_hpa': checked['pressure_hpa'], 'resolution': list(RESOLUTION)},
        'environment': {
            'python': platform.python_version(), 'platform': platform.platform(),
            'logical_cpu_count': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'dependencies': {'numpy': persistence.np.__version__, 'scipy': persistence.scipy.__version__,
                             'pandas': persistence.pd.__version__, 'gudhi': persistence.gd.__version__},
            'thread_environment': {name: os.environ.get(name) for name in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        },
        'native_build': native, 'timed_dates': args.dates, 'timing_repeats': args.repeats,
    }
    methodology = {
        'timer': 'time.perf_counter; wall-clock seconds',
        'filesystem_cache': 'Warm: preflight parses the input and hashing reads source/input files before measurements; no filesystem cache flush',
        'original': 'Complete unmodified AST-extracted compute_data with original file/date helpers',
        'direct_fresh': 'Fresh direct engine, no disk cache: constructor through first compute_data return',
        'planar_fresh': 'Fresh planar engine, no disk cache: constructor through first compute_data return',
        'planar_disk_hit': 'Fresh planar engine using the already populated temporary persistent cache: constructor through first compute_data return',
        'direct_warm_different_day': 'One complete query on an existing direct engine after an untimed query for a distinct date; no disk cache',
        'planar_warm_different_day': 'One complete query on an existing planar engine after an untimed query for a distinct date; no disk cache',
        'planar_cli_disk_hit': 'Wall time for a fresh Python process with explicit --method planar and --cache-dir, including imports, startup, cache loading, calculation, JSON output, process exit and parent JSON parsing',
        'cache_creation': 'Separately timed constructor and first planar query into an empty temporary cache; includes input parsing and geometry preparation; compared to the first timed original for the same date',
        'cache_scope': 'TemporaryDirectory is cleaned after all trials; disk files hold prepared data/geometry, never completed daily results; no user cache paths appear in report',
        'order': 'Rotate start among the seven cases by repetition index; cache creation and every trial run sequentially with no concurrent jobs started by the benchmark',
        'excluded_from_in_process': ['dependency imports', 'process startup', 'native compilation',
                                     'benchmark preflight', 'hashing', 'original AST extraction',
                                     'comparison checks', 'report serialization', 'analyzer destruction',
                                     'GUI', 'plotting', 'wind data'],
        'native_build': 'Explicit prebuild is required and excluded from all timings; cache creation loads the prebuilt native library before trials',
        'exactness': 'Every complete returned [depths, split, displacement] list must equal the same-repeat original exactly; CLI date also checked; signed zeros compare equal',
        'diagram_validation': 'This latency benchmark checks complete returned scores; run compare_daily.py --validation-only --method planar separately for full daily diagram validation',
        'speedup_scope': 'Original ratios use the in-process original median; the CLI ratio additionally includes its documented process/startup/JSON overhead. Planar-versus-direct ratios compare matching fresh or warm in-process scopes',
        'exit_requirement': 'All repetitions, cache creation, original repeat consistency, input immutability and source immutability must pass',
    }
    with tempfile.TemporaryDirectory(prefix='superpov-cpu-benchmark-') as directory:
        runs, creation = run_timing_trials(
            original, analyzer_class, [str(args.file)], RESOLUTION, args.dates,
            checked['dates'], args.repeats, Path(directory), pressure=checked['pressure_hpa'])
    input_unchanged = profile_cpu.warm_file_and_hash(args.file) == input_hash
    sources_unchanged = source_hashes() == hashes
    verification = {'requirement': 'exact_numeric', 'input_unchanged': input_unchanged,
                    'sources_unchanged': sources_unchanged,
                    'passed': report_passed(runs, creation, input_unchanged, sources_unchanged)}
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report = {'schema_version': 1, 'metadata': metadata, 'methodology': methodology,
              'cache_creation': creation, 'timing_runs': runs,
              'summary_by_date': summarize_timings(runs), 'verification': verification}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'result_file': args.output.name, 'verification': verification}), flush=True)
    if not verification['passed']:
        raise SystemExit('CPU comparison failed exact verification; see the result JSON')


if __name__ == '__main__':
    main()
