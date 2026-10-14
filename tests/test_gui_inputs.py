"""Regressions for date windows and optional wind input, without a display."""

import ast
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'SuPerPoV3.1.0.py'


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)


def input_namespace():
    nodes = ast.parse(SOURCE.read_text()).body
    namespace = {'datetime': datetime, 'timedelta': timedelta, 'np': np, 'pd': pd,
                 'Path': Path}
    execute([node for node in nodes if isinstance(node, ast.ImportFrom)
             and node.module == 'superpov_inputs'], namespace)
    return nodes, namespace


def value(item):
    return SimpleNamespace(get=lambda: item, focus_force=Mock())


class DateWindowTests(unittest.TestCase):
    def setUp(self):
        nodes, namespace = input_namespace()
        execute([node for node in nodes if isinstance(node, ast.FunctionDef)
                 and node.name in ('n_days_prior', 'n_days_after')], namespace)
        self.before = namespace['n_days_prior']
        self.after = namespace['n_days_after']

    def test_windows_clip_to_winter_and_exclude_center(self):
        cases = [
            ('1965-11-01', 31, 2, [], ['1965-11-02', '1965-11-03']),
            ('1965-11-02', 3, 0, ['1965-11-01'], []),
            ('1966-04-29', 0, 3, [], ['1966-04-30']),
            ('1966-04-30', 2, 31, ['1966-04-28', '1966-04-29'], []),
        ]
        for date, before, after, prior, later in cases:
            with self.subTest(date=date):
                self.assertEqual(self.before(date, before), prior)
                self.assertEqual(self.after(date, after), later)

    def test_year_transition_and_leap_day_use_calendar_math(self):
        self.assertEqual(self.before('1966-01-01', 2), ['1965-12-30', '1965-12-31'])
        self.assertEqual(self.after('1965-12-31', 2), ['1966-01-01', '1966-01-02'])
        self.assertEqual(self.after('1968-02-28', 2), ['1968-02-29', '1968-03-01'])
        self.assertEqual(self.before('1968-03-01', 2), ['1968-02-28', '1968-02-29'])
        self.assertEqual(self.after('1966-02-28', 1), ['1966-03-01'])

    def test_zero_returns_empty(self):
        for date in ('1965-11-01', '1966-01-01', '1966-04-30'):
            self.assertEqual(self.before(date, 0), [])
            self.assertEqual(self.after(date, 0), [])

    def test_huge_counts_are_clipped_before_date_arithmetic(self):
        prior = self.before('1968-03-01', 10 ** 100)
        later = self.after('1968-03-01', 10 ** 100)
        self.assertEqual((len(prior), prior[0], prior[-1]),
                         (121, '1967-11-01', '1968-02-29'))
        self.assertEqual((len(later), later[0], later[-1]),
                         (60, '1968-03-02', '1968-04-30'))

    def test_invalid_counts_and_dates_raise_clear_errors(self):
        for function in (self.before, self.after):
            for count in (-1, 1.5, True, '2.5', ''):
                with self.subTest(function=function.__name__, count=count):
                    with self.assertRaisesRegex(ValueError, 'non-negative integer'):
                        function('1965-12-01', count)
            for date in ('1966-05-01', '1966-02-29', '1966-2-01', '19660201', 'bad'):
                with self.subTest(function=function.__name__, date=date):
                    with self.assertRaisesRegex(ValueError, 'November|YYYY-MM-DD'):
                        function(date, 0)


class SubmitInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def submit(self, date='1965-11-01', before='0', after='0', wind=None,
               picked_file='', input_dict=None):
        nodes, namespace = input_namespace()
        ask = next(node for node in nodes if isinstance(node, ast.FunctionDef)
                   and node.name == 'ask_date_GUI')
        submit = next(node for node in ask.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'on_submit')
        namespace.update({
            'entry': value(date), 'entry_before': value(before), 'entry_after': value(after),
            'plot_var': value(10), 'press_vars': {10: value(True)},
            'guess_var': value('default'), 'lat_var': value(4), 'long_var': value(8),
            'output_var': value('D'), 'path_var': value(self.directory.name),
            'input_dict': {} if input_dict is None else input_dict,
            'messagebox': Mock(), 'root': Mock(),
            'filedialog': SimpleNamespace(askopenfilename=Mock(return_value=picked_file)),
            'tk': SimpleNamespace(TclError=RuntimeError, StringVar=lambda **kwargs: value(kwargs['value'])),
        })
        if wind is None:
            execute([node for node in ask.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'wind_var'
                             for target in node.targets)], namespace)
        else:
            namespace['wind_var'] = value(wind)
        execute([submit], namespace)
        namespace['on_submit']()
        return namespace

    def write_wind(self, content, name='u1060.txt'):
        path = Path(self.directory.name) / name
        path.write_text(content)
        return path

    def assert_rejected(self, namespace, message):
        self.assertFalse(namespace['root'].destroy.called)
        self.assertFalse(namespace['root'].quit.called)
        self.assertNotIn('date', namespace['input_dict'])
        errors = namespace['messagebox'].showerror.call_args_list
        self.assertTrue(errors)
        self.assertRegex(str(errors[-1]), message)

    def test_invalid_date_or_count_keeps_input_window_open(self):
        for date, before, after, message in [
            ('1966-05-01', '0', '0', 'November.*April'),
            ('1966-02-29', '0', '0', 'YYYY-MM-DD'),
            ('1966-2-01', '0', '0', 'YYYY-MM-DD'),
            ('1965-11-01', '-1', '0', 'non-negative integer'),
            ('1965-11-01', '0', '-1', 'non-negative integer'),
            ('1965-11-01', '1.5', '0', 'non-negative integer'),
            ('1965-11-01', '', '0', 'non-negative integer'),
        ]:
            with self.subTest(date=date, before=before, after=after):
                self.assert_rejected(self.submit(date, before, after, wind='F'), message)

    def test_default_does_not_require_or_open_wind_file(self):
        self.write_wind('this deliberately malformed file must be ignored')
        namespace = self.submit()
        self.assertTrue(namespace['root'].destroy.called)
        self.assertEqual(namespace['input_dict']['Wind'], 'F')
        self.assertEqual(namespace['input_dict']['date'], datetime(1965, 11, 1))
        self.assertEqual(namespace['input_dict']['daysBefore'], 0)
        self.assertEqual(namespace['input_dict']['daysAfter'], 0)
        self.assertFalse(namespace['filedialog'].askopenfilename.called)

    def test_enabled_wind_is_loaded_from_data_folder_before_close(self):
        path = self.write_wind('DATES\tSPEEDS\n1965-11-01\t-2\n1965-11-02\t3\n')
        namespace = self.submit(wind='E')
        self.assertTrue(namespace['root'].destroy.called)
        self.assertEqual(namespace['input_dict'].get('WindFile'), str(path))
        frame = namespace['input_dict'].get('WindData')
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame['DATES'].tolist(), ['1965-11-01', '1965-11-02'])
        self.assertEqual(frame['SPEEDS'].tolist(), [-2., 3.])
        self.assertFalse(namespace['filedialog'].askopenfilename.called)

    def test_enabled_wind_can_use_a_chosen_file(self):
        path = self.write_wind('DATES\tSPEEDS\n1965-11-01\t-2\n', name='chosen.tsv')
        namespace = self.submit(wind='E', picked_file=str(path))
        self.assertTrue(namespace['root'].destroy.called)
        self.assertEqual(namespace['input_dict'].get('WindFile'), str(path))
        self.assertIsInstance(namespace['input_dict'].get('WindData'), pd.DataFrame)

    def test_cancelled_or_missing_wind_file_keeps_input_window_open(self):
        self.assert_rejected(self.submit(wind='E'), 'Wind|wind')
        self.assert_rejected(self.submit(wind='E', picked_file=str(Path(self.directory.name) / 'missing.txt')),
                             'Wind|wind')

    def test_malformed_wind_file_keeps_input_window_open(self):
        for content in (
            '', 'date\tspeed\n1965-11-01\t-2\n',
            'DATES\tSPEEDS\n1965-11-01\tnot-a-number\n',
            'DATES\tSPEEDS\n1966-02-29\t-2\n',
            'DATES\tSPEEDS\n1965-11-01\tNaN\n',
            'DATES\tSPEEDS\n1965-11-01\tinf\n',
            'DATES\tSPEEDS\n',
        ):
            with self.subTest(content=content):
                self.write_wind(content)
                self.assert_rejected(self.submit(wind='E'), 'Wind|wind')

    def test_extra_wind_fields_are_rejected_before_closing(self):
        self.write_wind('DATES\tSPEEDS\n100\t1965-11-01\t-2\n')
        self.assert_rejected(self.submit(wind='E'), 'Wind|wind')

    def test_boolean_wind_speeds_are_rejected_before_closing(self):
        self.write_wind('DATES\tSPEEDS\n1965-11-01\tTrue\n1965-11-02\tFalse\n')
        self.assert_rejected(self.submit(wind='E'), 'Wind|wind')


if __name__ == '__main__':
    unittest.main()
