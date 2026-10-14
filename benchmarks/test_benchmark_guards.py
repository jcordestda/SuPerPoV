"""Focused regressions for benchmark validation and failure reporting.

Run: python -m unittest discover -s benchmarks -p 'test_*.py'
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import compare_persistence as comparison
import profile_cpu
from input_validation import validate_frame, validate_output_path


def day_frame():
    return pd.DataFrame({
        '# date': ['1965-11-01'] * 8280,
        'lev': [10.] * 8280,
        'lat': np.repeat(np.arange(90, -1, -2), 180),
        'lon': np.tile(np.arange(0, 360, 2), 46),
        'data': np.full(8280, 30000.),
    })


class BenchmarkGuards(unittest.TestCase):
    def test_profile_instrumentation_tolerates_changes_before_reference_functions(self):
        source = profile_cpu.SOURCE_PATH.read_text()
        for prefix in ('', '# Additional GUI imports and validation\n\n' * 17):
            with self.subTest(prefix_lines=prefix.count('\n')):
                namespace = profile_cpu.compile_functions(prefix + source, {}, instrumented=True)
                self.assertTrue(callable(namespace['compute_data']))

    def test_profile_instrumentation_rejects_changed_reference_body(self):
        source = profile_cpu.SOURCE_PATH.read_text()
        # Moving the entire function is safe; inserting inside a timed stage
        # must still fail rather than silently misattribute its statements.
        changed = source.replace('def compute_data(date, txt_files, resolution):',
                                 'def compute_data(date, txt_files, resolution):\n' + '\n' * 5)
        self.assertNotEqual(changed, source)
        with self.assertRaises(ValueError):
            profile_cpu.compile_functions(changed, {}, instrumented=True)

    def test_numeric_equality_is_not_labelled_bit_exact(self):
        bars = [[(3., 1.)], [(2., 1.)]]
        result = comparison.compare((bars, comparison.scores(bars)),
                                    (bars, comparison.scores(bars)))
        self.assertIn('diagram_exact', result)
        self.assertIn('scores_exact', result)
        self.assertFalse(any('bit_exact' in key for key in result))

    def test_earlier_mismatch_still_fails_when_last_repeat_matches(self):
        good = [[(3., 1.)], [(2., 1.)]]
        bad = [[(4., 1.)], [(2., 1.)]]
        reference = (good, comparison.scores(good))
        mismatch = (bad, comparison.scores(bad))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            output = Path(directory) / 'output.json'
            day_frame().to_csv(data, sep='\t', index=False)
            argv = ['compare_persistence.py', '--file', str(data), '--output',
                    str(output), '--all', '--methods', 'extended', 'direct', '--repeats', '2']
            # Isolate the expensive topology calls; exercise real CLI/report logic.
            with patch('sys.argv', argv), patch.object(comparison, 'geometry', return_value=[]), \
                    patch.object(comparison, 'calculate', side_effect=[reference, mismatch, reference, reference]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    comparison.main()
            self.assertNotEqual(failure.exception.code, 0)
            report = json.loads(output.read_text())
            self.assertFalse(report['verification']['within_tolerance'])
            self.assertEqual(len(report['results'][0]['comparisons_by_repeat']), 2)

    def test_compare_rejects_output_that_would_overwrite_input_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            data.write_text('original input\n')
            argv = ['compare_persistence.py', '--file', str(data), '--output', str(data)]
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
                try:
                    comparison.main()
                except (Exception, SystemExit) as failure:
                    self.assertIsInstance(failure, SystemExit)
                else:
                    self.fail('Unsafe output path was accepted')
            self.assertEqual(data.read_text(), 'original input\n')

    def test_profile_rejects_existing_python_output(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            output = Path(directory) / 'other.py'
            data.write_text('input\n')
            output.write_text('print("keep")\n')
            argv = ['profile_cpu.py', '--file', str(data), '--output', str(output)]
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    profile_cpu.parse_args()
            self.assertEqual(output.read_text(), 'print("keep")\n')

    def test_profile_preflight_rejects_wrong_resolution_and_malformed_grids(self):
        valid = day_frame()
        cases = [('wrong_resolution', valid, ['46', '90'])]
        reversed_rows = valid.iloc[::-1].reset_index(drop=True)
        cases.append(('row_order', reversed_rows, ['46', '180']))
        mixed_levels = valid.copy()
        mixed_levels.loc[1, 'lev'] = 50.
        cases.append(('multiple_levels', mixed_levels, ['46', '180']))
        infinite = valid.copy()
        infinite.loc[1, 'data'] = np.inf
        cases.append(('nonfinite', infinite, ['46', '180']))
        for name, frame, resolution in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                data = Path(directory) / 'input.tsv'
                frame.to_csv(data, sep='\t', index=False)
                argv = ['profile_cpu.py', '--file', str(data), '--output',
                        str(Path(directory) / 'out.json'), '--dates', '1965-11-01',
                        '--resolution', *resolution]
                with patch('sys.argv', argv), patch.object(profile_cpu, 'compile_functions',
                        side_effect=RuntimeError('Timed computation was reached')):
                    try:
                        profile_cpu.main()
                    except (Exception, SystemExit) as failure:
                        self.assertIsInstance(failure, SystemExit)
                        self.assertIn('Invalid input', str(failure))
                    else:
                        self.fail('Malformed grid was accepted')

    def test_missing_nonconsecutive_and_unsafe_final_dates_are_rejected(self):
        first = day_frame()
        second = first.copy()
        second['# date'] = '1965-11-02'
        frame = pd.concat([first, second], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'absent'):
            validate_frame(frame, (46, 180), ['1965-11-03'])
        with self.assertRaisesRegex(ValueError, 'final date'):
            validate_frame(frame, (46, 180), ['1965-11-02'], original_lookup=True)
        # Reading the first complete day is safe even in a one-day file.
        self.assertEqual(validate_frame(first, (46, 180), ['1965-11-01'],
                                        original_lookup=True)['rows_per_day'], 8280)
        second['# date'] = '1965-11-03'
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            validate_frame(pd.concat([first, second], ignore_index=True), (46, 180))
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_frame(pd.concat([frame, first], ignore_index=True), (46, 180))

    def test_original_year_routing_rejects_january_start(self):
        frame = day_frame()
        frame['# date'] = '1966-01-01'
        with self.assertRaisesRegex(ValueError, 'November or December'):
            validate_frame(frame, (46, 180), ['1966-01-01'], original_lookup=True)

    def test_output_alias_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'input.tsv'
            alias = Path(directory) / 'out.json'
            data.write_text('keep')
            alias.symlink_to(data)
            with self.assertRaises(ValueError):
                validate_output_path(alias, [data])
            self.assertEqual(data.read_text(), 'keep')

    def test_tolerance_acceptance_does_not_claim_numeric_exactness(self):
        good = [[(3., 1.)], [(2., 1.)]]
        close = [[(3. + 1e-12, 1.)], [(2., 1.)]]
        check = comparison.compare((good, comparison.scores(good)),
                                   (close, comparison.scores(close)))
        tolerant = comparison.verification_status([check], require_exact=False)
        strict = comparison.verification_status([check], require_exact=True)
        self.assertTrue(tolerant['within_tolerance'])
        self.assertFalse(tolerant['exact'])
        self.assertTrue(tolerant['passed'])
        self.assertFalse(strict['passed'])

    def test_template_reuse_preserves_complete_bars_across_positive_negative_and_tied_fields(self):
        # A triangulated disk with one interior vertex (4), not a periodic grid.
        triangles = np.array([[0, 1, 4], [0, 3, 4], [1, 2, 4], [2, 4, 5],
                              [3, 4, 6], [4, 6, 7], [4, 5, 8], [4, 7, 8]])
        template = comparison.prepare_triangle_template(triangles)
        tree, _ = template
        pristine = sorted((tuple(simplex), value) for simplex, value in tree.get_filtration())
        fields = [
            [9., 9., 9., 9., 1., 9., 9., 9., 9.],
            [-17., -17., -17., -17., -25., -17., -17., -17., -17.],
            [6.] * 9,
            [3., 8., 5., 7., 2., 6., 4., 10., 9.],
            [9., 9., 9., 9., 1., 9., 9., 9., 9.],
        ]
        for values in fields:
            with self.subTest(values=values):
                actual = comparison.pairs(triangles, values, 'template_direct', template)
                fresh = comparison.pairs(triangles, values, 'direct')
                self.assertEqual(sorted(actual), sorted(fresh))
                if min(values) > 0:
                    extended = comparison.pairs(triangles, values, 'extended')
                    self.assertEqual(sorted(actual), sorted(extended))
                self.assertEqual(sorted((tuple(simplex), value)
                                        for simplex, value in tree.get_filtration()), pristine)


if __name__ == '__main__':
    unittest.main()
