"""Timing-scope and correctness regressions for the daily benchmark."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))


class VirtualClock:
    def __init__(self):
        self.seconds = 0.

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


def valid_frame():
    blocks = []
    for date in ('1965-11-01', '1965-11-02'):
        blocks.append(pd.DataFrame({
            '# date': [date] * 8280,
            'lev': np.full(8280, 10.),
            'lat': np.repeat(np.arange(90, -1, -2), 180),
            'lon': np.tile(np.arange(0, 360, 2), 46),
            'data': np.full(8280, 30000.),
        }))
    return pd.concat(blocks, ignore_index=True)


class DailyBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = importlib.import_module('compare_daily')

    def test_fresh_trial_includes_constructor_and_warm_trial_uses_another_day(self):
        clock = VirtualClock()
        output = [[2.], 0., .5]

        class Analyzer:
            def __init__(self, **kwargs):
                clock.advance(10.)
                self.dates = []

            def compute_data(self, date, files, pressure=None):
                if date in self.dates:
                    raise AssertionError('A timed result-cache hit was permitted')
                self.dates.append(date)
                clock.advance(1.)
                return output

        def original(date, files, resolution):
            clock.advance(5.)
            return output

        with contextlib.redirect_stdout(io.StringIO()):
            runs = self.benchmark.run_timing_trials(
                original, Analyzer, ['input.tsv'], (46, 180), ['1965-11-01'],
                ['1965-11-01', '1965-11-02'], 2, pressure=10., clock=clock)
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(run['seconds'], {
                'original': 5., 'fresh_engine_first_query': 11.,
                'warm_engine_different_day': 1.})
            self.assertEqual(run['warmup_date'], '1965-11-02')
            self.assertTrue(run['verification']['passed'])
        self.assertNotEqual(runs[0]['execution_order'], runs[1]['execution_order'])

    def test_a_matching_final_repetition_cannot_hide_an_earlier_mismatch(self):
        clock = VirtualClock()
        expected = [[2.], 0., .5]

        class Analyzer:
            wrong_pending = True

            def __init__(self, **kwargs):
                pass

            def compute_data(self, date, files, pressure=None):
                clock.advance(1.)
                if date == '1965-11-01' and Analyzer.wrong_pending:
                    Analyzer.wrong_pending = False
                    return [[3.], 0., .5]
                return expected

        def original(*args):
            clock.advance(1.)
            return expected

        with contextlib.redirect_stdout(io.StringIO()):
            runs = self.benchmark.run_timing_trials(
                original, Analyzer, ['input.tsv'], (46, 180), ['1965-11-01'],
                ['1965-11-01', '1965-11-02'], 2, clock=clock)
        self.assertFalse(runs[0]['verification']['passed'])
        self.assertTrue(runs[1]['verification']['passed'])
        self.assertFalse(self.benchmark.report_passed(runs, None))

    def test_warm_measurement_requires_a_distinct_seed_date(self):
        with self.assertRaisesRegex(ValueError, 'different|distinct'):
            self.benchmark.run_timing_trials(
                None, None, [], (46, 180), ['1965-11-01'], ['1965-11-01'], 1)

    def test_full_validation_checks_grid_bars_even_when_scores_agree(self):
        frame = valid_frame()
        reference_bars = [[(3., 1.)], [(2., 1.)]]
        reference_scores = {'depths': [2.], 'split': 0., 'disp': .5, 'matched': (2., 1.)}
        candidate_bars = [reference_bars[0], [(2., 1.), (1.5, 1.4)]]

        class Analyzer:
            def analyze(self, *args, **kwargs):
                return candidate_bars, reference_scores

        with patch.object(self.benchmark.persistence, 'geometry', return_value=[]), \
                patch.object(self.benchmark.persistence, 'calculate',
                             return_value=(reference_bars, reference_scores)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.benchmark.validate_all_days(
                frame, Analyzer(), ['input.tsv'], pressure=10.)
        self.assertEqual(result['days'], 2)
        self.assertEqual(result['diagrams'], 4)
        self.assertEqual(result['bars'], 4)
        self.assertFalse(result['passed'])
        self.assertTrue(all(row['comparison']['scores_exact'] for row in result['by_date']))
        self.assertFalse(all(row['comparison']['diagram_exact'] for row in result['by_date']))

    def test_input_output_alias_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            data.write_text('keep input')
            with patch('sys.argv', ['compare_daily.py', '--file', str(data),
                                   '--output', str(data)]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.benchmark.main()
            self.assertEqual(data.read_text(), 'keep input')

    def test_failed_verification_writes_diagnostic_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            destination = Path(directory) / 'report.json'
            valid_frame().to_csv(data, sep='\t', index=False)
            argv = ['compare_daily.py', '--file', str(data), '--dates', '1965-11-01',
                    '--repeats', '1', '--output', str(destination)]
            failing = {'date': '1965-11-01', 'repeat': 1,
                       'seconds': {case: 1. for case in self.benchmark.CASES},
                       'outputs': {'original': [[2.], 0., .5]},
                       'verification': {'passed': False}}
            with patch('sys.argv', argv), \
                    patch.object(self.benchmark, 'load_analyzer_class', return_value=object), \
                    patch.object(self.benchmark, 'source_hashes', return_value={}), \
                    patch.object(self.benchmark, 'run_timing_trials', return_value=[failing]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    self.benchmark.main()
            self.assertNotEqual(failure.exception.code, 0)
            report = json.loads(destination.read_text())
            self.assertFalse(report['verification']['passed'])
            self.assertNotIn(directory, destination.read_text())

    def test_extended_method_is_used_by_both_candidate_timing_cases(self):
        output = [[2.], 0., .5]

        class Analyzer:
            def __init__(self, resolution, method):
                self.method = method

            def compute_data(self, *args, **kwargs):
                return output if self.method == 'extended' else [[3.], 0., .5]

        with contextlib.redirect_stdout(io.StringIO()):
            runs = self.benchmark.run_timing_trials(
                lambda *args: output, Analyzer, ['input.tsv'], (46, 180),
                ['1965-11-01'], ['1965-11-01', '1965-11-02'], 1, method='extended')
        self.assertTrue(runs[0]['verification']['passed'])

    def test_validation_only_has_no_original_calls_and_records_selected_method(self):
        bars = [[(3., 1.)], [(2., 1.)]]
        scores = {'depths': [2.], 'split': 0., 'disp': .5, 'matched': (2., 1.)}

        class Analyzer:
            def __init__(self, resolution, method):
                self.method = method

            def analyze(self, *args, **kwargs):
                return bars, scores

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            output = Path(directory) / 'report.json'
            valid_frame().to_csv(data, sep='\t', index=False)
            argv = ['compare_daily.py', '--file', str(data), '--output', str(output),
                    '--validation-only', '--method', 'extended']
            with patch('sys.argv', argv), \
                    patch.object(self.benchmark, 'load_analyzer_class', return_value=Analyzer), \
                    patch.object(self.benchmark, 'source_hashes', return_value={}), \
                    patch.object(self.benchmark.profile_cpu, 'compile_functions',
                                 side_effect=AssertionError('Original calculation was requested')), \
                    patch.object(self.benchmark.persistence, 'geometry', return_value=[]), \
                    patch.object(self.benchmark.persistence, 'calculate', return_value=(bars, scores)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.benchmark.main()
            report = json.loads(output.read_text())
        self.assertTrue(report['verification']['passed'])
        self.assertEqual(report['metadata']['method'], 'extended')
        self.assertEqual(report['timing_runs'], [])
        self.assertEqual(report['summary_by_date'], {})
        self.assertFalse(report['validation']['original_bridge_performed'])

    def test_original_bridge_mismatch_fails_an_otherwise_matching_validation(self):
        bars = [[(3., 1.)], [(2., 1.)]]
        scores = {'depths': [2.], 'split': 0., 'disp': .5, 'matched': (2., 1.)}

        class Analyzer:
            def analyze(self, *args, **kwargs):
                return bars, scores

        with patch.object(self.benchmark.persistence, 'geometry', return_value=[]), \
                patch.object(self.benchmark.persistence, 'calculate', return_value=(bars, scores)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.benchmark.validate_all_days(
                valid_frame(), Analyzer(), ['input.tsv'],
                original_outputs={'1965-11-01': [[4.], 0., .5]})
        self.assertFalse(result['passed'])
        self.assertTrue(all(row['passed'] for row in result['by_date']))
        self.assertFalse(result['original_bridge'][0]['extended_helper_equals_original_exactly'])


if __name__ == '__main__':
    unittest.main()
