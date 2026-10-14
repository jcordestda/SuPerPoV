#!/usr/bin/env python3
"""Compare complete original calls with fresh and reused DailyAnalyzer instances.

Examples, from the repository root:
    python benchmarks/compare_daily.py --file gph_winter_1965.txt --validate-all
    python benchmarks/compare_daily.py --file gph_winter_1965.txt --validation-only

Fresh-engine timing includes construction, input loading and validation, date
indexing, geometry, template preparation, and the first daily calculation. Warm
timing follows an untimed query for a different date on the same engine. Both
use warm filesystem caches; neither includes imports, GUI, or process startup.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import compare_persistence as persistence
from input_validation import validate_frame, validate_output_path
import profile_cpu


ROOT = Path(__file__).resolve().parents[1]
RESOLUTION = persistence.RES
DEFAULT_DATES = ('1965-11-01', '1965-12-23', '1966-04-30')
CASES = ('original', 'fresh_engine_first_query', 'warm_engine_different_day')


def load_analyzer_class():
    """Allow direct script execution without importing the GUI application."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from superpov_compute import DailyAnalyzer
    return DailyAnalyzer


def source_hashes():
    """Identify the actual source files without publishing machine paths."""
    paths = [profile_cpu.SOURCE_PATH, ROOT / 'superpov_compute.py', Path(__file__),
             Path(profile_cpu.__file__), Path(persistence.__file__),
             ROOT / 'benchmarks' / 'fast_build.py',
             ROOT / 'benchmarks' / 'input_validation.py']
    return {path.name: profile_cpu.warm_file_and_hash(path) for path in paths}


def run_timing_trials(original, analyzer_factory, txt_files, resolution, dates,
                      available_dates, repeats, pressure=None, clock=time.perf_counter,
                      method='direct'):
    """Time calls sequentially and retain every original/candidate comparison."""
    if len(set(available_dates)) < 2:
        raise ValueError('Warm measurements require a distinct, different seed date')
    preferred_seeds = [date for date in available_dates if date not in dates]
    first_original = {}
    runs = []
    for date in dates:
        seed = next(day for day in (preferred_seeds or available_dates) if day != date)
        for repeat in range(repeats):
            start = repeat % len(CASES)
            order = CASES[start:] + CASES[:start]
            elapsed = {}
            outputs = {}
            for case in order:
                if case == 'warm_engine_different_day':
                    analyzer = analyzer_factory(resolution=resolution, method=method)
                    analyzer.compute_data(seed, txt_files, pressure=pressure)
                    started = clock()
                    output = analyzer.compute_data(date, txt_files, pressure=pressure)
                    elapsed[case] = clock() - started
                    del analyzer
                elif case == 'fresh_engine_first_query':
                    started = clock()
                    analyzer = analyzer_factory(resolution=resolution, method=method)
                    output = analyzer.compute_data(date, txt_files, pressure=pressure)
                    elapsed[case] = clock() - started
                    del analyzer
                else:
                    started = clock()
                    output = original(date, txt_files, resolution)
                    elapsed[case] = clock() - started
                outputs[case] = profile_cpu.json_value(output)
            first_original.setdefault(date, outputs['original'])
            checks = {
                'fresh_engine_exact': outputs['fresh_engine_first_query'] == outputs['original'],
                'warm_engine_exact': outputs['warm_engine_different_day'] == outputs['original'],
                'original_repeat_exact': outputs['original'] == first_original[date],
            }
            checks['passed'] = all(checks.values())
            runs.append({'date': date, 'repeat': repeat + 1,
                         'execution_order': list(order), 'warmup_date': seed,
                         'seconds': elapsed, 'outputs': outputs, 'verification': checks})
            print(json.dumps({'date': date, 'repeat': repeat + 1,
                              'seconds': elapsed, 'verification': checks}), flush=True)
    return runs


def summarize_timings(runs):
    summary = {}
    for date in dict.fromkeys(run['date'] for run in runs):
        selected = [run for run in runs if run['date'] == date]
        cases = {case: profile_cpu.summary([run['seconds'][case] for run in selected])
                 for case in CASES}
        baseline = cases['original']['median_seconds']
        summary[date] = {
            'cases': cases,
            'speedup_from_medians': {
                case: baseline / cases[case]['median_seconds']
                for case in CASES if case != 'original'},
        }
    return summary


def validate_all_days(frame, analyzer, txt_files, pressure=None, original_outputs=None):
    """Check full diagrams against the existing extended algorithm on every day.

    Only data extraction and geometry preparation are shared across reference
    days; each extended calculation still builds fresh original-style trees.
    These are correctness calls, not latency samples.
    """
    triangles = persistence.geometry()
    by_date = []
    bridges = []
    original_outputs = original_outputs or {}
    for date, group in frame.groupby('# date', sort=False):
        reference = persistence.calculate(triangles, group['data'].to_numpy(), 'extended')
        actual = analyzer.analyze(date, txt_files, pressure=pressure)
        if len(reference[0]) != 2 or len(actual[0]) != 2:
            raise ValueError('Daily validation requires both polar and grid diagrams')
        comparison = persistence.compare(reference, actual)
        passed = comparison['diagram_exact'] and comparison['scores_exact']
        by_date.append({'date': date, 'comparison': comparison, 'passed': passed})
        if date in original_outputs:
            scores = reference[1]
            legacy = profile_cpu.json_value([scores['depths'], scores['split'], scores['disp']])
            bridges.append({'date': date,
                            'extended_helper_equals_original_exactly': legacy == original_outputs[date]})
        print(json.dumps({'validation_date': date, 'passed': passed,
                          'bar_counts': comparison['bar_counts']}), flush=True)
    bridge_passed = all(row['extended_helper_equals_original_exactly'] for row in bridges)
    return {
        'reference': 'Existing extended construction with prepared daily arrays and geometry',
        'candidate': 'DailyAnalyzer.analyze using the selected persistence method and reused caches',
        'days': len(by_date), 'diagrams': 2 * len(by_date),
        'bars': sum(sum(row['comparison']['bar_counts']) for row in by_date),
        'by_date': by_date, 'original_bridge': bridges,
        'original_bridge_performed': bool(bridges),
        'passed': bool(by_date) and bridge_passed and all(row['passed'] for row in by_date),
    }


def report_passed(runs, validation):
    return (bool(runs or validation is not None)
            and all(run['verification']['passed'] for run in runs)
            and (validation is None or validation['passed']))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True, help='One 46 x 180 single-pressure winter TSV')
    parser.add_argument('--dates', nargs='+', help='Timed dates; default: November 1, December 23, April 30 of 1965/66')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--method', choices=('direct', 'extended'), default='direct',
                        help='Candidate persistence method; default: direct')
    parser.add_argument('--validate-all', action='store_true', help='Also compare both full diagrams on every input date')
    parser.add_argument('--validation-only', action='store_true', help='Validate every input date without timing or original-function calls')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'benchmarks' / 'results' / 'daily_comparison.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    if args.validation_only and args.dates:
        parser.error('--dates selects timed queries; omit it with --validation-only')
    if not args.file.is_file():
        parser.error(f'Input file does not exist: {args.file}')
    args.file = args.file.resolve()
    try:
        args.output = validate_output_path(
            args.output, (args.file, profile_cpu.SOURCE_PATH, ROOT / 'superpov_compute.py', Path(__file__)))
    except ValueError as error:
        parser.error(str(error))
    args.dates = [] if args.validation_only else (args.dates or list(DEFAULT_DATES))
    return args


def main():
    args = parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        # Benchmark preflight is untimed. The candidate receives the file path,
        # never this dataframe, and must perform its own first-query preparation.
        frame = persistence.pd.read_csv(args.file, sep='\t')
        checked = validate_frame(frame, RESOLUTION, args.dates or None,
                                 original_lookup=not args.validation_only)
        if not args.validation_only and len(checked['dates']) < 2:
            raise ValueError('Warm measurements require a distinct, different seed date')
    except (ValueError, OSError) as error:
        raise SystemExit(f'Invalid input: {error}') from error
    input_hash = profile_cpu.warm_file_and_hash(args.file)
    hashes = source_hashes()
    analyzer_class = load_analyzer_class()
    metadata = {
        'started_at_utc': started_utc,
        'source_sha256': hashes,
        'input': {'name': args.file.name, 'sha256': input_hash,
                  'bytes': args.file.stat().st_size, 'days': len(checked['dates']),
                  'date_range': [checked['dates'][0], checked['dates'][-1]],
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
        'timed_dates': args.dates, 'timing_repeats': 0 if args.validation_only else args.repeats,
        'method': args.method,
    }
    methodology = {
        'timer': 'time.perf_counter; wall-clock seconds',
        'filesystem_cache': 'Warm: benchmark preflight parses the input and hashing reads it before measurements; no cache flush',
        'original': 'Unmodified AST-extracted compute_data and its original file/date helpers',
        'fresh_engine_first_query': 'Constructor through first compute_data return; includes loading, validation, date indexing, geometry, templates, and daily computation',
        'warm_engine_different_day': 'Complete compute_data call after an untimed different-date query on the same analyzer; includes file fingerprint checks and daily computation',
        'cache_scope': 'New analyzer for each case and repetition; instance data and geometry caches; no result caching',
        'order': 'Rotate the start among the three timing cases by repetition index; all calls are sequential',
        'warmup': 'No topology warmup for fresh engines; preflight warms parser and filesystem cache',
        'excluded': ['dependency imports', 'process startup', 'GUI', 'plotting', 'wind data',
                     'benchmark preflight', 'hashing', 'original AST extraction',
                     'comparison checks', 'separate all-day validation', 'output serialization',
                     'analyzer destruction after returned results'],
        'exactness': 'Numeric equality of full returned lists and full sorted diagram endpoints; signed zeros compare equal',
        'exit_requirement': 'Exact numeric agreement for every timed repetition and every requested validation date',
        'validation_only_note': 'A validation-only invocation does not perform an original-function bridge or report latency',
    }
    files = [str(args.file)]
    runs = []
    if not args.validation_only:
        dependencies = {'pd': persistence.pd, 'np': persistence.np,
                        'gd': persistence.gd, 'Delaunay': persistence.Delaunay}
        original = profile_cpu.compile_functions(profile_cpu.SOURCE_PATH.read_text(), dependencies)['compute_data']
        runs = run_timing_trials(original, analyzer_class, files, RESOLUTION, args.dates,
                                 checked['dates'], args.repeats, pressure=checked['pressure_hpa'],
                                 method=args.method)
    validation = None
    if args.validate_all or args.validation_only:
        original_outputs = {}
        for run in runs:
            original_outputs.setdefault(run['date'], run['outputs']['original'])
        analyzer = analyzer_class(resolution=RESOLUTION, method=args.method)
        validation = validate_all_days(frame, analyzer, files, checked['pressure_hpa'], original_outputs)
    input_unchanged = profile_cpu.warm_file_and_hash(args.file) == input_hash
    verification = {'requirement': 'exact_numeric', 'input_unchanged': input_unchanged,
                    'passed': input_unchanged and report_passed(runs, validation)}
    metadata['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report = {'schema_version': 1, 'metadata': metadata, 'methodology': methodology,
              'timing_runs': runs, 'summary_by_date': summarize_timings(runs),
              'validation': validation, 'verification': verification}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'result_file': args.output.name, 'verification': verification}), flush=True)
    if not verification['passed']:
        raise SystemExit('Daily comparison failed exact verification; see the result JSON')


if __name__ == '__main__':
    main()
