"""Native planar persistence checked against the existing GUDHI calculation."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial import Delaunay

from superpov_compute import DailyAnalyzer

try:
    import superpov_planar as planar
except ModuleNotFoundError as error:
    if error.name != 'superpov_planar':
        raise
    planar = None


ROOT = Path(__file__).resolve().parents[1]


def can_build_native():
    if sys.platform not in ('linux', 'darwin'):
        return False
    configured = os.environ.get('CXX')
    if configured:
        command = shlex.split(configured)
        return bool(command and shutil.which(command[0]))
    return any(shutil.which(name) for name in ('c++', 'g++', 'clang++'))


def meshes(resolution):
    rows, columns = resolution
    r, theta = np.meshgrid(np.linspace(.001, 4., rows),
                           np.linspace(0., 2 * np.pi, columns, endpoint=False), indexing='ij')
    polar = np.column_stack(((r * np.cos(theta)).ravel(), (r * np.sin(theta)).ravel()))
    x, y = np.meshgrid(np.linspace(0., 90., rows),
                      np.linspace(0., 720., 2 * columns, endpoint=False), indexing='ij')
    grid = np.column_stack((x.ravel(), y.ravel()))
    return [Delaunay(points).simplices for points in (polar, grid)]


class OptionalBackendTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planar, 'The planar persistence backend has not been implemented')

    def test_native_test_class_skips_without_a_compiler_or_prebuilt_library(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'tests').mkdir()
            for name in ('superpov_planar.py', 'superpov_planar.cpp', 'superpov_compute.py'):
                shutil.copyfile(ROOT / name, directory / name)
            shutil.copyfile(Path(__file__), directory / 'tests' / 'test_planar.py')
            result = subprocess.run(
                [sys.executable, '-m', 'unittest', 'tests.test_planar.PlanarTests.'
                 'test_interior_minimum_is_a_finite_reflected_h1_pair', '-v'],
                cwd=directory, env=dict(os.environ, CXX='/compiler-that-does-not-exist'),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('skipped=1', result.stderr)

    def test_unbuilt_import_is_safe_and_never_invokes_a_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ('superpov_planar.py', 'superpov_planar.cpp'):
                shutil.copyfile(ROOT / name, directory / name)
            compiler = directory / 'compiler.py'
            compiler.write_text("from pathlib import Path\n"
                                "Path(__file__).with_suffix('.ran').write_text('invoked')\n")
            environment = dict(os.environ, CXX=shlex.join([sys.executable, str(compiler)]))
            code = ("import superpov_planar as p; "
                    "\nif p.native_available(): raise AssertionError('unbuilt backend available')"
                    "\ntry: p.PlanarComplex([[0, 1, 2]], 3)"
                    "\nexcept p.NativeUnavailable: pass"
                    "\nelse: raise AssertionError('unbuilt backend was accepted')")
            result = subprocess.run([sys.executable, '-c', code], cwd=directory, env=environment,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(compiler.with_suffix('.ran').exists(), 'Import started a compiler')
            self.assertFalse(list(directory.glob('__pycache__/*.so')))
            self.assertFalse(list(directory.glob('__pycache__/*.dylib')))
            environment['CXX'] = '/compiler-that-does-not-exist'
            failed_build = subprocess.run([sys.executable, 'superpov_planar.py', '--build'],
                                          cwd=directory, env=environment, capture_output=True, text=True)
            self.assertNotEqual(failed_build.returncode, 0)
            diagnostic = 'compiler' if sys.platform in ('linux', 'darwin') else 'supports'
            self.assertIn(diagnostic, failed_build.stderr.lower())
            self.assertNotIn('Traceback', failed_build.stderr)


class PlanarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if planar is None:
            return
        if not can_build_native():
            if planar.native_available():
                return
            raise unittest.SkipTest('Optional native backend needs Linux/macOS and a C++ compiler '
                                    'or a usable prebuilt library')
        result = subprocess.run([sys.executable, str(ROOT / 'superpov_planar.py'), '--build'],
                                capture_output=True, text=True)
        if result.returncode:
            raise AssertionError('Explicit native build failed: ' + result.stderr)

    def setUp(self):
        self.assertIsNotNone(planar, 'The planar persistence backend has not been implemented')

    def test_interior_minimum_is_a_finite_reflected_h1_pair(self):
        triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        complex_ = planar.PlanarComplex(triangles, 5)
        self.assertEqual(complex_.pairs([8., 8., 8., 8., -3.]), [(8., -3.)])
        self.assertEqual(complex_.pairs([8., 8., 8., 8., 9.]), [])
        self.assertEqual(complex_.pairs([8.] * 5), [])

    def test_both_original_meshes_match_direct_for_float64_ties_and_negative_values(self):
        for resolution in ((2, 3), (4, 8), (9, 17)):
            direct = DailyAnalyzer(resolution, method='direct')
            direct._prepare_geometry()
            triangulations = meshes(resolution)
            complexes = [planar.PlanarComplex(triangles, np.prod(resolution) * factor)
                         for triangles, factor in zip(triangulations, (1, 2))]
            rng = np.random.default_rng(317)
            fields = [rng.normal(-32., 15., np.prod(resolution)) for _ in range(5)]
            fields += [rng.integers(-3, 4, np.prod(resolution)).astype(float),
                       np.full(np.prod(resolution), -17.),
                       rng.normal(0., 1e-5, np.prod(resolution))]
            for field in fields:
                doubled = np.hstack((field.reshape(resolution), field.reshape(resolution))).ravel()
                for index, (values, complex_) in enumerate(zip((field, doubled), complexes)):
                    with self.subTest(resolution=resolution, mesh=index, field=field):
                        expected = direct._pairs(direct._templates[index], values)
                        actual = complex_.pairs(values)
                        self.assertEqual(sorted(actual), sorted(expected))
                        self.assertEqual([a - b for a, b in actual],
                                         sorted((a - b for a, b in actual), reverse=True))

    def test_persistence_cutoff_is_strict_and_preserves_float64_values(self):
        complex_ = planar.PlanarComplex([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], 5)
        cutoff = 1e-5
        for high, expected in ((np.nextafter(cutoff, 0.), []), (cutoff, []),
                               (np.nextafter(cutoff, np.inf), [(np.nextafter(cutoff, np.inf), 0.)])):
            self.assertEqual(complex_.pairs([high] * 4 + [0.]), expected)
        self.assertEqual(complex_.pairs([2.] * 4 + [1.], min_persistence=1.), [])
        self.assertEqual(complex_.pairs([2.] * 4 + [1.], min_persistence=0.), [(2., 1.)])

    def test_repeated_and_concurrent_days_have_no_shared_mutable_state(self):
        triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        complex_ = planar.PlanarComplex(triangles, 5)
        triangles[:] = 0
        fields = [np.array([9., 9., 9., 9., 2.]), np.ones(5), np.array([4., 4., 4., 4., -2.])]
        with ThreadPoolExecutor(max_workers=3) as executor:
            actual = list(executor.map(complex_.pairs, fields * 10))
        self.assertEqual(actual, [[(9., 2.)], [], [(4., -2.)]] * 10)
        np.testing.assert_array_equal(fields[0], [9., 9., 9., 9., 2.])

    def test_invalid_meshes_and_values_fail_before_native_access(self):
        for triangles, count in (([[0, 1, 2], [0, 1, 3], [0, 1, 4]], 5),
                                 ([[0, 1, 2], [2, 0, 1]], 3),
                                 ([[0, 0, 1]], 2), ([[0, 1, -1]], 3),
                                 ([[0, 1, 3]], 3), ([[0., 1.5, 2.]], 3),
                                 ([[0, 1]], 3), ([], 3), ([[0, 1, 2]], 2.5)):
            with self.subTest(triangles=triangles, count=count), self.assertRaises(ValueError):
                planar.PlanarComplex(triangles, count)
        complex_ = planar.PlanarComplex([[0, 1, 2]], 3)
        for values in ([1., 2.], [1., 2., np.nan], [1., 2., np.inf], [[1., 2., 3.]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                complex_.pairs(values)
        for cutoff in (-1., np.inf, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                complex_.pairs([1., 2., 3.], min_persistence=cutoff)

    def test_a_source_change_does_not_load_an_obsolete_native_library(self):
        if not can_build_native():
            self.skipTest('Source invalidation test requires a C++ compiler on Linux/macOS')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ('superpov_planar.py', 'superpov_planar.cpp'):
                shutil.copyfile(ROOT / name, directory / name)
            result = subprocess.run([sys.executable, 'superpov_planar.py', '--build'],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, '-c',
                                     'import superpov_planar as p; '
                                     '\nif not p.native_available(): raise AssertionError("build unavailable")'],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            source = directory / 'superpov_planar.cpp'
            source.write_text(source.read_text() + '\n// A new native source version.\n')
            result = subprocess.run([sys.executable, '-c',
                                     'import superpov_planar as p; '
                                     '\nif p.native_available(): raise AssertionError("stale build available")'],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
