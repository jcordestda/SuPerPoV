"""Optional native persistence for SuPerPoV's triangulated planar disks.

Build explicitly with ``python superpov_planar.py --build``. Importing this
module never starts a compiler. The application can use ``native_available``
to choose its GUDHI fallback on machines without a built native library.
"""
import ctypes
import hashlib
import operator
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np


class NativeUnavailable(RuntimeError):
    """The optional native backend has not been built or cannot be loaded."""


_LIBRARY = None


def _library_path():
    source = Path(__file__).resolve().with_suffix('.cpp')
    if sys.platform not in ('linux', 'darwin'):
        raise NativeUnavailable('Native planar persistence supports Linux and macOS; use GUDHI direct')
    try:
        contents = source.read_bytes()
    except OSError as error:
        raise NativeUnavailable('Native planar source is unavailable: ' + str(error)) from error
    identity = (sys.platform, platform.machine(), ctypes.sizeof(ctypes.c_void_p), sys.byteorder)
    digest = hashlib.sha256(contents + repr(identity).encode()).hexdigest()[:20]
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    return source.parent / '__pycache__' / ('superpov_planar_' + digest + suffix)


def build_native():
    """Compile the current source explicitly, publishing the library atomically."""
    global _LIBRARY
    destination = _library_path()
    configured = os.environ.get('CXX')
    try:
        compiler = shlex.split(configured) if configured else [
            next((name for name in ('c++', 'g++', 'clang++') if shutil.which(name)), '')]
    except ValueError as error:
        raise NativeUnavailable('Invalid CXX compiler command: ' + str(error)) from error
    if not compiler or not compiler[0] or not shutil.which(compiler[0]):
        raise NativeUnavailable('No C++ compiler available; set CXX or use GUDHI direct')
    source = Path(__file__).resolve().with_suffix('.cpp')
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp',
                                                dir=destination.parent)
        os.close(descriptor)
        shared = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
        result = subprocess.run(compiler + ['-std=c++11', '-O3', '-DNDEBUG', '-fPIC', shared,
                                            str(source), '-o', temporary],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise NativeUnavailable('Native planar compiler failed: ' + result.stderr.strip())
        os.replace(temporary, destination)
    except OSError as error:
        raise NativeUnavailable('Cannot build native planar library: ' + str(error)) from error
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    _LIBRARY = None
    return destination


def _load_native():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = _library_path()
    try:
        library = ctypes.CDLL(str(path))
        function = library.superpov_planar_pairs
        integer_pointer = ctypes.POINTER(ctypes.c_int64)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        function.argtypes = [ctypes.c_int64, ctypes.c_int64, integer_pointer, integer_pointer,
                             integer_pointer, double_pointer, ctypes.c_double, double_pointer]
        function.restype = ctypes.c_int64
    except (OSError, AttributeError) as error:
        raise NativeUnavailable('Native planar library is unavailable; run '
                                '`python superpov_planar.py --build` or use GUDHI direct') from error
    _LIBRARY = library
    return library


def native_available():
    """Whether the current native source has a usable, already built library."""
    try:
        _load_native()
    except NativeUnavailable:
        return False
    return True


class PlanarComplex:
    """Cache the dual adjacency of a filled triangulated planar disk.

    The caller supplies the original triangulation, not a replacement grid.
    Interior holes and nonplanar surfaces are outside this specialized API.
    Arrays retained here are immutable; each ``pairs`` call has independent
    native working memory. Equal lifetimes may have a different order from
    GUDHI, so a caller depending on a particular tied bar must use GUDHI.
    """

    def __init__(self, triangles, vertex_count):
        try:
            vertex_count = operator.index(vertex_count)
        except TypeError as error:
            raise ValueError('vertex_count must be a positive integer') from error
        if vertex_count < 3:
            raise ValueError('vertex_count must be at least three')
        triangles = np.asarray(triangles)
        if (triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0
                or not np.issubdtype(triangles.dtype, np.integer)):
            raise ValueError('triangles must be a nonempty integer array with shape (n, 3)')
        if np.any(triangles < 0) or np.any(triangles >= vertex_count):
            raise ValueError('triangle vertex indices must lie within vertex_count')
        triangles = np.array(triangles, dtype=np.int64, order='C', copy=True)
        canonical = np.sort(triangles, axis=1)
        if np.any(canonical[:, 1:] == canonical[:, :-1]):
            raise ValueError('triangles must have three distinct vertices')
        if len(np.unique(canonical, axis=0)) != len(triangles):
            raise ValueError('duplicate triangles are not allowed')

        # Group the original primal edges while retaining their incident faces.
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        edges.sort(axis=1)
        face_ids = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, face_ids = edges[order], face_ids[order]
        starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
        counts = np.diff(np.r_[starts, len(edges)])
        if np.any(counts > 2):
            raise ValueError('nonmanifold edge has more than two incident triangles')
        neighbors = np.empty((len(starts), 2), dtype=np.int64)
        neighbors[:, 0] = face_ids[starts]
        neighbors[:, 1] = len(triangles)  # One common exterior face.
        interior = counts == 2
        neighbors[interior, 1] = face_ids[starts[interior] + 1]
        self._triangles = triangles
        self._edges = np.ascontiguousarray(edges[starts])
        self._neighbors = neighbors
        for array in (self._triangles, self._edges, self._neighbors):
            array.setflags(write=False)
        self.vertex_count = vertex_count
        self._native = _load_native()

    def pairs(self, values, min_persistence=1e-5):
        """Return finite H1 of negated heights reflected as (high, low) pairs."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.vertex_count,) or not np.isfinite(values).all():
            raise ValueError('values must be a finite one-dimensional array matching vertex_count')
        try:
            minimum = float(min_persistence)
        except (TypeError, ValueError) as error:
            raise ValueError('min_persistence must be finite and nonnegative') from error
        if not np.isfinite(minimum) or minimum < 0.:
            raise ValueError('min_persistence must be finite and nonnegative')
        values = np.ascontiguousarray(values)
        output = np.empty((len(self._triangles), 2), dtype=np.float64)
        integer_pointer = ctypes.POINTER(ctypes.c_int64)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        count = self._native.superpov_planar_pairs(
            len(self._triangles), len(self._edges),
            self._triangles.ctypes.data_as(integer_pointer),
            self._edges.ctypes.data_as(integer_pointer),
            self._neighbors.ctypes.data_as(integer_pointer),
            values.ctypes.data_as(double_pointer), minimum,
            output.ctypes.data_as(double_pointer))
        if count < 0 or count > len(output):
            raise RuntimeError('Native planar persistence failed to allocate working memory')
        return [tuple(pair) for pair in output[:count]]


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True,
                        help='Compile the optional C++ backend with CXX or the system compiler')
    parser.parse_args()
    try:
        print(build_native())
    except NativeUnavailable as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
