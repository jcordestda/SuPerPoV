"""Integration checks for persistent data/geometry and optional CPU backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import superpov_compute as computation
from test_daily_analyzer import DATES, RESOLUTION, write_winter, original_calculation


class AcceleratedAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'winter.tsv'
        self.cache = self.root / 'cache'
        self.values = write_winter(self.path)

    def analyzer(self, method='direct'):
        return computation.DailyAnalyzer(RESOLUTION, method=method, cache_dir=self.cache)

    def test_new_analyzer_reuses_disk_data_and_geometry(self):
        first = self.analyzer().compute_data(DATES[0], [self.path])
        with patch.object(pd, 'read_csv', side_effect=AssertionError('TSV reparse')), \
                patch.object(computation, 'Delaunay', side_effect=AssertionError('Retriangulation')):
            second = self.analyzer()
            self.assertEqual(first, second.compute_data(DATES[0], [self.path]))
            np.testing.assert_array_equal(second.get_values(DATES[1], [self.path]), self.values[1])
        self.assertEqual(first, original_calculation()(DATES[0], [str(self.path)], RESOLUTION))

    def test_rewritten_source_invalidates_disk_entry_even_with_restored_mtime(self):
        self.analyzer().get_values(DATES[0], [self.path])
        stat = self.path.stat()
        changed = write_winter(self.path, seed=9)
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        np.testing.assert_array_equal(self.analyzer().get_values(DATES[0], [self.path]), changed[0])

    def test_utf8_bom_header_preserves_original_file_support(self):
        self.path.write_bytes(b'\xef\xbb\xbf' + self.path.read_bytes())
        expected = original_calculation()(DATES[0], [str(self.path)], RESOLUTION)
        self.assertEqual(self.analyzer().compute_data(DATES[0], [self.path]), expected)
        self.assertEqual(self.analyzer().compute_data(DATES[0], [self.path]), expected)

    def test_same_basename_pressure_and_resolution_do_not_share_cache(self):
        self.analyzer().get_values(DATES[0], [self.path], pressure=10)
        other = self.root / '50hPa' / self.path.name
        values = write_winter(other, pressure=50, seed=8)
        np.testing.assert_array_equal(self.analyzer().get_values(DATES[0], [other], pressure=50), values[0])
        with self.assertRaisesRegex(ValueError, 'pressure'):
            self.analyzer().get_values(DATES[0], [other], pressure=10)
        with self.assertRaisesRegex(ValueError, 'grid|resolution'):
            computation.DailyAnalyzer((4, 7), cache_dir=self.cache).get_values(DATES[0], [self.path])

    def test_corrupt_or_unavailable_disk_cache_does_not_prevent_analysis(self):
        expected = self.analyzer().compute_data(DATES[0], [self.path])
        for entry in self.cache.glob('*.npz'):
            entry.write_bytes(b'truncated cache')
        self.assertEqual(self.analyzer().compute_data(DATES[0], [self.path]), expected)
        unavailable = self.root / 'not_a_directory'
        unavailable.write_text('occupied')
        analyzer = computation.DailyAnalyzer(RESOLUTION, cache_dir=unavailable)
        self.assertEqual(analyzer.compute_data(DATES[0], [self.path]), expected)

    def test_auto_falls_back_when_optional_native_backend_missing(self):
        from superpov_planar import NativeUnavailable
        with patch('superpov_planar.native_available', return_value=False):
            analyzer = self.analyzer(method='auto')
            self.assertEqual(analyzer.method, 'direct')
            self.assertEqual(analyzer.compute_data(DATES[0], [self.path]),
                             self.analyzer().compute_data(DATES[0], [self.path]))
            with self.assertRaises(NativeUnavailable):
                self.analyzer(method='planar')

    def test_planar_disk_hit_new_process_avoids_heavy_optional_imports(self):
        from superpov_planar import native_available
        if not native_available():
            self.skipTest('Build optional CPU backend with python superpov_planar.py --build')
        expected = self.analyzer('planar').compute_data(DATES[0], [self.path])
        script = '''import json, sys
from superpov_compute import DailyAnalyzer
result = DailyAnalyzer((4, 8), method='planar', cache_dir=sys.argv[2]).compute_data(sys.argv[3], [sys.argv[1]])
print(json.dumps({'result': result, 'heavy': [name for name in ('pandas', 'scipy', 'gudhi') if name in sys.modules]}))
'''
        run = subprocess.run([sys.executable, '-c', script, str(self.path), str(self.cache), DATES[0]],
                             capture_output=True, text=True, check=True)
        actual = json.loads(run.stdout)
        self.assertEqual(actual['result'], expected)
        self.assertEqual(actual['heavy'], [])

    def test_tied_leading_polar_features_preserve_direct_displacement(self):
        from superpov_planar import native_available
        if not native_available():
            self.skipTest('Build optional CPU backend with python superpov_planar.py --build')
        # The raw native order selects death=0 (displacement=1); GUDHI selects
        # death=1 (displacement=0). Both leading features have lifetime=2.
        field = [0, 0, 4, 1, 2, 5, 1, 4, 0, 0, 0, 1, 2, 0, 3, 4,
                 3, 0, 4, 0, 3, 5, 2, 2, 1, 1, 1, 0, 4, 2, 5, 2]
        frame = pd.read_csv(self.path, sep='\t')
        frame['data'] = np.tile(field, len(DATES))
        frame.to_csv(self.path, sep='\t', index=False)
        expected = self.analyzer('direct').analyze(DATES[0], [self.path])
        actual = self.analyzer('planar').analyze(DATES[0], [self.path])
        self.assertEqual(expected[1]['disp'], 0.)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
