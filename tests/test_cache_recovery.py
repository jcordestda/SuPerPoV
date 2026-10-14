"""Valid NPZ corruption must rebuild caches before any persistence backend runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import superpov_compute as computation
from superpov_planar import native_available
from test_daily_analyzer import DATES, RESOLUTION, original_calculation, write_winter


def rewrite_entry(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    change(arrays)
    np.savez(path, **arrays)


class CacheRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'winter.tsv'
        # The original GUI calculation requires a following date block.
        self.values = write_winter(self.path, dates=[*DATES, '1965-11-04'])
        original = original_calculation()
        self.expected = [original(day, [str(self.path)], RESOLUTION) for day in DATES]

    def test_geometry_faults_recover_on_first_and_second_process_for_each_backend(self):
        methods = ['auto', 'direct', 'extended']
        if native_available():
            methods.append('planar')

        def duplicate_face(arrays):
            arrays['polar'][0] = arrays['polar'][1]

        def incomplete_mesh(arrays):
            arrays['polar'] = np.array([[0, 1, 2]], dtype=arrays['polar'].dtype)

        for method in methods:
            for fault in (duplicate_face, incomplete_mesh):
                with self.subTest(method=method, fault=fault.__name__):
                    cache = self.root / (method + '-' + fault.__name__)
                    computation.DailyAnalyzer(RESOLUTION, method=method, cache_dir=cache).compute_data(
                        DATES[0], [self.path])
                    entry, = cache.glob('geometry-*.npz')
                    rewrite_entry(entry, fault)
                    broken = entry.read_bytes()
                    command = [sys.executable, str(Path(computation.__file__)),
                               '--file', str(self.path), '--dates', *DATES,
                               '--resolution', '4', '8', '--method', method,
                               '--cache-dir', str(cache)]
                    for attempt in range(2):
                        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        rows = [json.loads(line) for line in run.stdout.splitlines()]
                        self.assertEqual([row['date'] for row in rows], DATES)
                        actual = [[row['depths'], row['split'], row['displacement']] for row in rows]
                        self.assertEqual(actual, self.expected)
                        if attempt == 0:
                            repaired = entry.read_bytes()
                            self.assertTrue(repaired != broken, 'Corrupt geometry was not replaced')
                        else:
                            self.assertEqual(entry.read_bytes(), repaired)

    def test_changed_finite_winter_values_are_rebuilt_then_reused(self):
        cache = self.root / 'winter-cache'
        computation.DailyAnalyzer(RESOLUTION, cache_dir=cache).get_values(DATES[0], [self.path])
        entry, = cache.glob('winter-*.npz')

        def change(arrays):
            arrays['values'][0, 0] += 100.

        rewrite_entry(entry, change)
        rebuilt = computation.DailyAnalyzer(RESOLUTION, cache_dir=cache)
        np.testing.assert_array_equal(rebuilt.get_values(DATES[0], [self.path]), self.values[0])
        with patch.object(pd, 'read_csv', side_effect=AssertionError('TSV reparse')):
            reused = computation.DailyAnalyzer(RESOLUTION, cache_dir=cache)
            np.testing.assert_array_equal(reused.get_values(DATES[0], [self.path]), self.values[0])

    def test_legacy_geometry_and_winter_are_rebuilt_then_reused(self):
        cache = self.root / 'legacy-cache'
        computation.DailyAnalyzer(RESOLUTION, cache_dir=cache).compute_data(DATES[0], [self.path])

        def make_legacy(arrays):
            current = json.loads(arrays['__metadata__'].tobytes())
            legacy = {'schema_version': 1, 'metadata': current['metadata']}
            arrays['__metadata__'] = np.frombuffer(json.dumps(legacy).encode(), dtype=np.uint8)

        entries = list(cache.glob('*.npz'))
        self.assertEqual(len(entries), 2)
        for entry in entries:
            rewrite_entry(entry, make_legacy)
        legacy_bytes = {entry: entry.read_bytes() for entry in entries}
        rebuilt = computation.DailyAnalyzer(RESOLUTION, cache_dir=cache)
        self.assertEqual(rebuilt.compute_data(DATES[0], [self.path]), self.expected[0])
        for entry in entries:
            self.assertTrue(entry.read_bytes() != legacy_bytes[entry], 'Legacy entry was not replaced')
        with patch.object(pd, 'read_csv', side_effect=AssertionError('TSV reparse')), \
                patch.object(computation, 'Delaunay', side_effect=AssertionError('Retriangulation')):
            reused = computation.DailyAnalyzer(RESOLUTION, cache_dir=cache)
            self.assertEqual(reused.compute_data(DATES[0], [self.path]), self.expected[0])


if __name__ == '__main__':
    unittest.main()
