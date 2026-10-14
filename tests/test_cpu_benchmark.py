"""Correctness and artifact-safety guards for the final CPU comparison."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
EXPECTED = [[2., 1.], .25, .5]
DATES = ['1965-11-01', '1965-11-02']


class AnalyzerFixture:
    """Cheap stand-in for expensive topology; cache IO remains real."""

    def __init__(self, resolution, method, cache_dir=None):
        self.method = method
        self.queried = []
        self.cache_dir = cache_dir

    def compute_data(self, date, files, pressure=None):
        if date in self.queried:
            raise AssertionError('A timed query reused an already computed date')
        self.queried.append(date)
        if self.cache_dir is not None:
            Path(self.cache_dir, 'prepared.bin').write_bytes(b'prepared winter')
        return EXPECTED


def cli_fixture(date, files, resolution, cache_dir, pressure=None):
    return {'date': date, 'depths': EXPECTED[0], 'split': EXPECTED[1],
            'displacement': EXPECTED[2], 'elapsed_seconds': .01}


def input_frame():
    return pd.DataFrame({
        '# date': np.repeat(DATES, 8280), 'lev': np.full(16560, 10.),
        'lat': np.tile(np.repeat(np.arange(90, -1, -2), 180), 2),
        'lon': np.tile(np.arange(0, 360, 2), 92), 'data': np.full(16560, 30000.),
    })


class CPUBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = importlib.import_module('compare_cpu')

    def run_trials(self, directory, analyzer=AnalyzerFixture, original=None,
                   cli=cli_fixture, repeats=2):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.benchmark.run_timing_trials(
                original or (lambda *args: EXPECTED), analyzer, ['input.tsv'],
                (46, 180), [DATES[0]], DATES, repeats, Path(directory),
                pressure=10., cli_runner=cli)

    def test_every_trial_checks_original_and_distinct_warm_date(self):
        with tempfile.TemporaryDirectory() as directory:
            runs, creation = self.run_trials(directory)
        self.assertEqual(len(runs), 2)
        self.assertTrue(creation['verification']['passed'])
        self.assertGreater(creation['bytes'], 0)
        self.assertTrue(self.benchmark.report_passed(runs, creation, True, True))
        self.assertNotEqual(runs[0]['execution_order'], runs[1]['execution_order'])
        for run in runs:
            self.assertEqual(run['warmup_date'], DATES[1])
            self.assertEqual(set(run['seconds']), set(self.benchmark.CASES))
            self.assertTrue(all(seconds >= 0 for seconds in run['seconds'].values()))
            self.assertEqual(run['outputs']['planar_cli_disk_hit'], EXPECTED)

    def test_later_matching_repeat_does_not_hide_candidate_mismatch(self):
        class OnceWrong(AnalyzerFixture):
            wrong_pending = True

            def compute_data(self, *args, **kwargs):
                result = super().compute_data(*args, **kwargs)
                if self.method == 'direct' and OnceWrong.wrong_pending:
                    OnceWrong.wrong_pending = False
                    return [[99.], .25, .5]
                return result

        with tempfile.TemporaryDirectory() as directory:
            runs, creation = self.run_trials(directory, analyzer=OnceWrong)
        self.assertFalse(runs[0]['verification']['direct_fresh_exact'])
        self.assertTrue(runs[1]['verification']['passed'])
        self.assertFalse(self.benchmark.report_passed(runs, creation, True, True))

    def test_candidates_compare_to_the_same_original_repeat(self):
        values = iter([EXPECTED, [[99.], .25, .5]])
        with tempfile.TemporaryDirectory() as directory:
            runs, creation = self.run_trials(directory, original=lambda *args: next(values))
        self.assertTrue(runs[0]['verification']['passed'])
        self.assertFalse(runs[1]['verification']['direct_fresh_exact'])
        self.assertFalse(runs[1]['verification']['original_repeat_exact'])
        self.assertFalse(self.benchmark.report_passed(runs, creation, True, True))

    def test_warm_direct_case_uses_a_direct_engine(self):
        class WrongWarmDirect(AnalyzerFixture):
            def compute_data(self, *args, **kwargs):
                result = super().compute_data(*args, **kwargs)
                if self.method == 'direct' and len(self.queried) > 1:
                    return [[99.], .25, .5]
                return result

        with tempfile.TemporaryDirectory() as directory:
            runs, creation = self.run_trials(directory, analyzer=WrongWarmDirect, repeats=1)
        self.assertTrue(runs[0]['verification']['direct_fresh_exact'])
        self.assertTrue(runs[0]['verification']['planar_warm_different_day_exact'])
        self.assertFalse(runs[0]['verification']['direct_warm_different_day_exact'])
        self.assertFalse(self.benchmark.report_passed(runs, creation, True, True))

    def test_malformed_compiler_environment_does_not_block_metadata(self):
        with patch.dict(os.environ, {'CXX': '"'}):
            metadata = self.benchmark.native_build_metadata()
        self.assertIsNone(metadata['compiler_version_available_at_benchmark'])

    def test_cli_checks_every_score_and_date(self):
        for key, wrong in [('depths', [2., 99.]), ('split', 99.),
                           ('displacement', 99.), ('date', DATES[1])]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                def wrong_cli(*args, **kwargs):
                    payload = cli_fixture(*args, **kwargs)
                    payload[key] = wrong
                    return payload

                runs, creation = self.run_trials(directory, cli=wrong_cli, repeats=1)
                self.assertFalse(runs[0]['verification']['planar_cli_disk_hit_exact'])
                self.assertFalse(self.benchmark.report_passed(runs, creation, True, True))

    def test_cache_creation_is_also_verified_against_original(self):
        class WrongCreation(AnalyzerFixture):
            pending = True

            def compute_data(self, *args, **kwargs):
                result = super().compute_data(*args, **kwargs)
                if self.cache_dir is not None and WrongCreation.pending:
                    WrongCreation.pending = False
                    return [[99.], .25, .5]
                return result

        with tempfile.TemporaryDirectory() as directory:
            runs, creation = self.run_trials(directory, analyzer=WrongCreation)
        self.assertTrue(all(run['verification']['passed'] for run in runs))
        self.assertFalse(creation['verification']['passed'])
        self.assertFalse(self.benchmark.report_passed(runs, creation, True, True))

    def test_subprocess_receives_explicit_planar_and_cache_arguments(self):
        script = '''import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--file', nargs='+', required=True)
p.add_argument('--dates', nargs='+', required=True)
p.add_argument('--resolution', nargs=2, required=True)
p.add_argument('--method', choices=['planar'], required=True)
p.add_argument('--cache-dir', required=True)
p.add_argument('--pressure', type=float, required=True)
a = p.parse_args()
assert a.pressure == 10. and a.resolution == ['46', '180']
assert len(a.dates) == 1
print(json.dumps({'date': a.dates[0], 'depths': [2., 1.], 'split': .25,
                  'displacement': .5, 'elapsed_seconds': .01}))
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'cli.py')
            path.write_text(script)
            with patch.object(self.benchmark, 'COMPUTE_PATH', path):
                payload = self.benchmark.run_cli(
                    DATES[0], ['input.tsv'], (46, 180), Path(directory), pressure=10.)
        self.assertEqual(payload, cli_fixture(DATES[0], [], (), None))

    def test_real_planar_cli_cache_hit_matches_complete_original_result(self):
        import superpov_planar

        if not superpov_planar.native_available():
            self.skipTest('Explicit native build required for the planar CLI integration check')
        resolution = (4, 8)
        days = [*DATES, '1965-11-03']
        values = 30000. + np.random.default_rng(3).integers(0, 2000, (3, 32))
        frame = pd.DataFrame({
            '# date': np.repeat(days, 32), 'lev': 10.,
            'lat': np.tile(np.repeat(np.linspace(90., 0., 4), 8), 3),
            'lon': np.tile(np.arange(0., 360., 45.), 12), 'data': values.ravel(),
        })
        dependencies = {'pd': pd, 'np': np, 'gd': self.benchmark.persistence.gd,
                        'Delaunay': self.benchmark.persistence.Delaunay}
        original = self.benchmark.profile_cpu.compile_functions(
            self.benchmark.profile_cpu.SOURCE_PATH.read_text(), dependencies)['compute_data']
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, 'winter.tsv')
            cache = Path(directory, 'cache')
            frame.to_csv(data, sep='\t', index=False)
            self.benchmark.load_analyzer_class()(resolution, method='planar', cache_dir=cache).compute_data(
                DATES[0], [data], pressure=10.)
            payload = self.benchmark.run_cli(DATES[1], [data], resolution, cache, pressure=10.)
            expected = original(DATES[1], [str(data)], resolution)
        self.assertEqual(payload['date'], DATES[1])
        self.assertEqual([payload['depths'], payload['split'], payload['displacement']], expected)
        self.assertGreaterEqual(payload['elapsed_seconds'], 0.)

    def test_input_and_source_output_aliases_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, 'input.tsv')
            data.write_text('keep input')
            alias = Path(directory, 'alias.json')
            alias.hardlink_to(data)
            for output in (data, alias, ROOT / 'superpov_planar.cpp'):
                with self.subTest(output=output.name), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.benchmark.parse_args(['--file', str(data), '--output', str(output)])
            self.assertEqual(data.read_text(), 'keep input')

    def test_input_mutation_is_reported_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, 'input.tsv')
            output = Path(directory, 'report.json')
            input_frame().to_csv(data, sep='\t', index=False)
            cache = Path(directory, 'cache')
            cache.mkdir()
            runs, creation = self.run_trials(cache, repeats=1)

            def mutate_input(*args, **kwargs):
                with data.open('a') as handle:
                    handle.write('\n')
                return runs, creation

            with patch.object(self.benchmark, 'load_analyzer_class', return_value=AnalyzerFixture), \
                    patch.object(self.benchmark, 'source_hashes', return_value={}), \
                    patch.object(self.benchmark, 'native_build_metadata', return_value={}), \
                    patch.object(self.benchmark, 'run_timing_trials', side_effect=mutate_input), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.benchmark.main(['--file', str(data), '--dates', DATES[0],
                                         '--repeats', '1', '--output', str(output)])
            report = json.loads(output.read_text())
            self.assertFalse(report['verification']['input_unchanged'])
            self.assertFalse(report['verification']['passed'])
            self.assertNotIn(directory, output.read_text())


if __name__ == '__main__':
    unittest.main()
