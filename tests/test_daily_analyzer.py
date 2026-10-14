"""Behavioral checks for cached file routing and daily persistence."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gudhi as gd
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

try:
    import superpov_compute as computation
except ModuleNotFoundError as error:
    if error.name != 'superpov_compute':
        raise
    computation = None


RESOLUTION = (4, 8)
DATES = ['1965-11-01', '1965-11-02', '1965-11-03']


def write_winter(path, pressure=10, seed=3, dates=DATES):
    values = 30000. + np.random.default_rng(seed).integers(0, 2000, (len(dates), 32))
    frame = pd.DataFrame({
        '# date': np.repeat(dates, 32), 'lev': pressure,
        'lat': np.tile(np.repeat(np.linspace(90., 0., 4), 8), len(dates)),
        'lon': np.tile(np.arange(0., 360., 45.), len(dates) * 4),
        'data': values.ravel(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False)
    return values


def original_calculation():
    source = Path(__file__).resolve().parents[1] / 'SuPerPoV3.1.0.py'
    tree = ast.parse(source.read_text())
    names = {'correct_df', 'date_index_returner', 'compute_data'}
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'pd': pd, 'np': np, 'gd': gd, 'Delaunay': Delaunay}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['compute_data']


class DailyAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(computation, 'The cached daily analyzer has not been implemented')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'winter.txt'
        self.values = write_winter(self.path)
        self.engine = computation.DailyAnalyzer(RESOLUTION)

    def test_first_interior_and_final_dates_use_correct_values(self):
        for index, date in enumerate(DATES):
            np.testing.assert_array_equal(self.engine.get_values(date, [self.path]), self.values[index])

    def test_only_one_full_parse_for_different_days_and_returns_independent_arrays(self):
        with patch.object(pd, 'read_csv', wraps=pd.read_csv) as read:
            exposed = self.engine.get_values(DATES[0], [self.path])
            exposed[:] = -1
            np.testing.assert_array_equal(self.engine.get_values(DATES[1], [self.path]), self.values[1])
            np.testing.assert_array_equal(self.engine.get_values(DATES[0], [self.path]), self.values[0])
        self.assertEqual(sum('nrows' not in call.kwargs for call in read.call_args_list), 1)

    def test_modified_same_size_file_is_reloaded_even_when_mtime_is_restored(self):
        self.engine.get_values(DATES[0], [self.path])
        previous = self.path.stat()
        changed = write_winter(self.path, seed=7)
        self.assertEqual(self.path.stat().st_size, previous.st_size)
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        np.testing.assert_array_equal(self.engine.get_values(DATES[0], [self.path]), changed[0])

    def test_atomic_replacement_is_reloaded(self):
        self.engine.get_values(DATES[0], [self.path])
        replacement = self.path.with_suffix('.new')
        changed = write_winter(replacement, seed=9)
        replacement.replace(self.path)
        np.testing.assert_array_equal(self.engine.get_values(DATES[0], [self.path]), changed[0])

    def test_pressure_files_with_same_basename_never_share_fields(self):
        other = self.path.parent / '50hPa' / self.path.name
        alternate = write_winter(other, pressure=50, seed=7)
        for path, pressure, values in ((self.path, 10, self.values), (other, 50, alternate),
                                       (self.path, 10, self.values)):
            np.testing.assert_array_equal(self.engine.get_values(DATES[0], [path], pressure=pressure), values[0])
        with self.assertRaisesRegex(ValueError, 'pressure'):
            self.engine.get_values(DATES[0], [self.path], pressure=50)

    def test_winter_routing_does_not_fully_parse_unneeded_files(self):
        other = self.path.parent / 'earlier.txt'
        write_winter(other, dates=['1964-11-01', '1964-11-02', '1964-11-03'])
        with patch.object(pd, 'read_csv', wraps=pd.read_csv) as read:
            np.testing.assert_array_equal(self.engine.get_values(DATES[0], [other, self.path]), self.values[0])
        fully_read = [Path(call.args[0]) for call in read.call_args_list if 'nrows' not in call.kwargs]
        self.assertEqual(fully_read, [self.path])

    def test_bounded_cache_reloads_evicted_file(self):
        other = self.path.parent / 'other.txt'
        write_winter(other, seed=5)
        engine = computation.DailyAnalyzer(RESOLUTION, max_cached_files=1)
        with patch.object(pd, 'read_csv', wraps=pd.read_csv) as read:
            for path in (self.path, other, self.path):
                engine.get_values(DATES[0], [path])
        self.assertEqual(sum('nrows' not in call.kwargs for call in read.call_args_list), 3)

    def test_geometry_reused_but_daily_results_recomputed_and_mutation_isolated(self):
        original = original_calculation()
        with patch.object(computation, 'Delaunay', wraps=Delaunay) as triangulate:
            result = self.engine.compute_data(DATES[0], [self.path])
            self.assertEqual(result, original(DATES[0], [str(self.path)], RESOLUTION))
            result[0][:] = [-1.]
            self.assertEqual(self.engine.compute_data(DATES[1], [self.path]),
                             original(DATES[1], [str(self.path)], RESOLUTION))
            self.assertEqual(self.engine.compute_data(DATES[0], [self.path]),
                             original(DATES[0], [str(self.path)], RESOLUTION))
        self.assertEqual(triangulate.call_count, 2)

    def test_missing_dates_and_wrong_grid_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'date'):
            self.engine.get_values('1965-11-04', [self.path])
        with self.assertRaisesRegex(ValueError, 'grid|resolution|rows'):
            computation.DailyAnalyzer((4, 7)).get_values(DATES[0], [self.path])
        frame = pd.read_csv(self.path, sep='\t')
        frame.loc[0, 'lon'] = 45
        frame.to_csv(self.path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'lon|grid'):
            self.engine.get_values(DATES[0], [self.path])

    def test_invalid_data_rejected_before_topology(self):
        for column, value in [('data', np.nan), ('lev', 50.), ('# date', '1965-11-02')]:
            with self.subTest(column=column):
                write_winter(self.path)
                frame = pd.read_csv(self.path, sep='\t')
                frame.loc[0, column] = value
                frame.to_csv(self.path, sep='\t', index=False)
                with patch.object(computation, 'Delaunay', wraps=Delaunay) as triangulate:
                    with self.assertRaises(ValueError):
                        self.engine.compute_data(DATES[0], [self.path])
                self.assertEqual(triangulate.call_count, 0)

    def test_headless_cli_outputs_scores_for_multiple_dates(self):
        command = [sys.executable, str(Path(computation.__file__)), '--file', str(self.path),
                   '--no-disk-cache', '--dates', DATES[0], DATES[1], '--resolution', '4', '8', '--pressure', '10']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([row['date'] for row in rows], DATES[:2])
        original = original_calculation()
        for row in rows:
            self.assertEqual([row['depths'], row['split'], row['displacement']],
                             original(row['date'], [str(self.path)], RESOLUTION))

    def test_degenerate_longitude_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'longitude|columns'):
            computation.DailyAnalyzer((4, 2))


if __name__ == '__main__':
    unittest.main()
