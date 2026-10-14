"""Cached daily SuPerPoV calculation, independent of the graphical interface.

One DailyAnalyzer shares validated winter arrays and fixed triangulations across
queries. It never caches completed scores or mutates a previously processed tree.
"""
from collections import OrderedDict
import csv
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
import operator
from pathlib import Path

import numpy as np


MIN_PERSISTENCE = 1e-5


def Delaunay(points):
    """Import triangulation only when no reusable geometry is available."""
    from scipy.spatial import Delaunay as triangulate
    return triangulate(points)


@lru_cache(maxsize=1)
def _geometry_versions():
    return {'numpy': np.__version__, 'scipy': version('scipy'), 'geometry_revision': 1}


def _date(value):
    try:
        parsed = datetime.strptime(value, '%Y-%m-%d').date()
    except (TypeError, ValueError) as error:
        raise ValueError('Expected a date in YYYY-MM-DD format') from error
    if parsed.isoformat() != value or parsed.month not in (11, 12, 1, 2, 3, 4):
        raise ValueError('Expected a November–April date in YYYY-MM-DD format')
    return parsed


def _winter_year(parsed):
    return parsed.year if parsed.month >= 11 else parsed.year - 1


def _fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class _Winter:
    values: np.ndarray
    date_index: dict
    pressure: float


class DailyAnalyzer:
    """Calculate dates with bounded in-memory file caches and shared geometry.

    Files must contain one pressure and one November–April winter, with complete
    north-to-south, eastward regular grids. Missing calendar days are allowed;
    requested dates must exist. The first file from the requested winter wins,
    preserving the application's ordered-file selection rule.

    ``method='auto'`` uses the optional compiled planar backend when built,
    otherwise ``direct`` computes finite ordinary H1 of negated heights.
    ``method='extended'`` retains extended persistence for numerical comparisons.
    ``cache_dir`` optionally reuses validated arrays and geometry across processes.
    Each instance is intended for one sequential analysis, not concurrent calls.
    """

    def __init__(self, resolution=(46, 180), method='auto', max_cached_files=8, cache_dir=None):
        try:
            self.resolution = tuple(operator.index(n) for n in resolution)
            self.max_cached_files = operator.index(max_cached_files)
        except (TypeError, ValueError) as error:
            raise ValueError('Resolution and cache size must be integers') from error
        if len(self.resolution) != 2 or min(self.resolution) < 2:
            raise ValueError('Resolution must contain two dimensions of at least 2')
        if self.resolution[1] < 3:
            raise ValueError('At least three longitude columns are needed for polar triangulation')
        if self.max_cached_files < 1:
            raise ValueError('The file cache must hold at least one winter')
        if method not in ('auto', 'planar', 'direct', 'extended'):
            raise ValueError('Persistence method must be auto, planar, direct or extended')
        if method in ('auto', 'planar'):
            from superpov_planar import NativeUnavailable, native_available
            available = native_available()
            if method == 'planar' and not available:
                raise NativeUnavailable('Run python superpov_planar.py --build before using --method planar')
            method = 'planar' if available else 'direct'
        self.method = method
        self._disk = None
        if cache_dir is not None:
            from superpov_cache import DiskCache
            self._disk = DiskCache(cache_dir)
        self._headers = OrderedDict()
        self._files = OrderedDict()
        self._templates = None
        self._triangles = None
        self._direct_templates = None

    def _select_file(self, date, txt_files):
        year = _winter_year(_date(date))
        for filename in txt_files:
            path = Path(filename).resolve()
            signature = _fingerprint(path)
            cached = self._headers.get(path)
            if cached is None or cached[0] != signature:
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    header = next(csv.DictReader(stream, delimiter='\t'), None)
                if not header or not header.get('# date'):
                    raise ValueError('Winter files must have a nonempty # date column')
                file_year = _winter_year(_date(header['# date']))
                if _fingerprint(path) != signature:
                    raise ValueError('Input file changed while reading its header; retry the query')
                self._headers[path] = (signature, file_year)
                if len(self._headers) > 256:
                    self._headers.popitem(last=False)
            else:
                file_year = cached[1]
                self._headers.move_to_end(path)
            if file_year == year:
                return path, signature, year
        raise ValueError('No input file contains the winter for date ' + date)

    def _read_winter(self, path, year):
        import pandas as pd
        frame = pd.read_csv(path, sep='\t')
        required = {'# date', 'lev', 'lat', 'lon', 'data'}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError('Expected nonempty TSV columns: # date, lev, lat, lon, data')
        for column in ('lev', 'lat', 'lon', 'data'):
            try:
                finite = np.isfinite(frame[column].to_numpy(dtype=np.float64)).all()
            except (TypeError, ValueError) as error:
                raise ValueError(column + ' must contain finite numbers') from error
            if not finite:
                raise ValueError(column + ' must contain finite numbers')
        if frame['lev'].nunique() != 1:
            raise ValueError('Expected exactly one pressure level per input file')
        row_dates = frame['# date'].to_numpy()
        starts = np.r_[0, np.flatnonzero(row_dates[1:] != row_dates[:-1]) + 1]
        ends = np.r_[starts[1:], len(frame)]
        size = self.resolution[0] * self.resolution[1]
        if np.any(ends - starts != size):
            raise ValueError('Each date must contain a complete grid matching the resolution')
        dates = [str(row_dates[start]) for start in starts]
        parsed = [_date(date) for date in dates]
        if any(_winter_year(date) != year for date in parsed):
            raise ValueError('Each input file must contain dates from one winter')
        if any(left >= right for left, right in zip(parsed, parsed[1:])):
            raise ValueError('Date blocks must be unique and chronological')
        expected_lat = np.repeat(np.linspace(90., 0., self.resolution[0]), self.resolution[1])
        expected_lon = np.tile(np.linspace(0., 360., self.resolution[1], endpoint=False), self.resolution[0])
        for column, expected in (('lat', expected_lat), ('lon', expected_lon)):
            actual = frame[column].to_numpy(dtype=np.float64).reshape(-1, size)
            if not np.allclose(actual, expected[None, :], rtol=0., atol=1e-10):
                raise ValueError(column + ' grid must be ordered north-to-south and eastward')
        values = frame['data'].to_numpy(dtype=np.float64, copy=True).reshape(len(dates), size)
        values.setflags(write=False)
        return _Winter(values, {date: index for index, date in enumerate(dates)},
                       float(frame['lev'].iloc[0]))

    def _cached_winter(self, arrays, year):
        """Validate the numeric cache representation before using it as input."""
        if arrays is None:
            return None
        try:
            values, ordinals, pressure = (arrays[key] for key in ('values', 'dates', 'pressure'))
            if (values.dtype != np.float64 or values.ndim != 2
                    or values.shape[1] != np.prod(self.resolution)
                    or ordinals.dtype != np.int64 or ordinals.shape != (len(values),)
                    or pressure.dtype != np.float64 or pressure.shape != (1,)
                    or not np.isfinite(pressure).all()
                    or not len(values) or not np.isfinite(values).all()):
                return None
            dates = [datetime.fromordinal(int(day)).date().isoformat() for day in ordinals]
            if (any(_winter_year(_date(day)) != year for day in dates)
                    or any(a >= b for a, b in zip(dates, dates[1:]))):
                return None
            return _Winter(values, {day: i for i, day in enumerate(dates)}, float(pressure[0]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def get_values(self, date, txt_files, pressure=None):
        """Return an independent float64 day array, loading its file only as needed."""
        path, signature, year = self._select_file(date, txt_files)
        cached = self._files.get(path)
        if cached is None or cached[0] != signature:
            metadata = {'fingerprint': list(signature), 'resolution': list(self.resolution),
                        'winter_year': year, 'validation_revision': 1,
                        'numpy': np.__version__, 'pandas': version('pandas')}
            winter = None
            if self._disk is not None:
                winter = self._cached_winter(self._disk.load('winter', str(path), metadata), year)
            needs_store = winter is None
            if winter is None:
                winter = self._read_winter(path, year)
            if _fingerprint(path) != signature:
                raise ValueError('Input file changed while reading; retry the query')
            if needs_store and self._disk is not None:
                self._disk.store('winter', str(path), metadata, {
                    'values': winter.values,
                    'dates': np.array([_date(day).toordinal() for day in winter.date_index], dtype=np.int64),
                    'pressure': np.array([winter.pressure], dtype=np.float64)})
            self._files[path] = (signature, winter)
            self._files.move_to_end(path)
            if len(self._files) > self.max_cached_files:
                self._files.popitem(last=False)
        else:
            winter = cached[1]
            self._files.move_to_end(path)
        if pressure is not None and winter.pressure != pressure:
            raise ValueError('Input pressure does not match the requested pressure')
        if date not in winter.date_index:
            raise ValueError('Requested date is absent from the selected winter: ' + date)
        return winter.values[winter.date_index[date]].copy()

    def _prepare_geometry(self):
        metadata = dict(_geometry_versions(), resolution=list(self.resolution))
        key = str(self.resolution)
        arrays = None if self._disk is None else self._disk.load('geometry', key, metadata)
        counts = (np.prod(self.resolution), 2 * np.prod(self.resolution))
        if arrays is not None:
            triangles = [arrays.get(name) for name in ('polar', 'grid')]
            if all(array is not None and array.dtype.kind in 'iu' and array.ndim == 2
                   and array.shape[1] == 3 and len(array) and array.min() >= 0
                   and array.max() < count for array, count in zip(triangles, counts)):
                self._triangles = triangles
        if self._triangles is None:
            self._triangles = self._triangulate()
            if self._disk is not None:
                self._disk.store('geometry', key, metadata,
                                 dict(zip(('polar', 'grid'), self._triangles)))
        if self.method == 'planar':
            from superpov_planar import PlanarComplex
            self._templates = [PlanarComplex(triangles, count)
                               for triangles, count in zip(self._triangles, counts)]
        else:
            self._templates = self._make_templates()

    def _triangulate(self):
        rows, columns = self.resolution
        r, theta = np.meshgrid(np.linspace(.001, 4., rows),
                               np.linspace(0., 2 * np.pi, columns, endpoint=False), indexing='ij')
        polar = np.column_stack(((r * np.cos(theta)).ravel(), (r * np.sin(theta)).ravel()))
        x, y = np.meshgrid(np.linspace(0., 90., rows),
                          np.linspace(0., 720., 2 * columns, endpoint=False), indexing='ij')
        grid = np.column_stack((x.ravel(), y.ravel()))
        return [Delaunay(points).simplices for points in (polar, grid)]

    def _make_templates(self):
        import gudhi as gd
        templates = []
        for triangles in self._triangles:
            vertices = np.unique(triangles)[None, :]
            tree = gd.SimplexTree()
            tree.insert_batch(triangles.T, np.full(len(triangles), np.inf))
            # Batch insertion lowers vertices; repair raises higher simplices.
            # This pristine template is intentionally not a valid filtration.
            tree.reset_filtration(-np.inf, min_dim=1)
            templates.append((tree, vertices))
        return templates

    def _pairs(self, template, values):
        if self.method == 'planar':
            return template.pairs(values, MIN_PERSISTENCE)
        return self._gudhi_pairs(template, values, self.method)

    @staticmethod
    def _gudhi_pairs(template, values, method):
        pristine, vertices = template
        tree = pristine.copy()
        filtration = -values if method == 'direct' else values
        tree.insert_batch(vertices, filtration[vertices[0]])
        tree.make_filtration_non_decreasing()
        if method == 'direct':
            tree.compute_persistence(homology_coeff_field=11, min_persistence=MIN_PERSISTENCE,
                                     persistence_dim_max=False)
            intervals = tree.persistence_intervals_in_dimension(1)
            if not np.isfinite(intervals).all():
                raise ValueError('Direct persistence returned nonfinite intervals')
            pairs = [tuple(-pair) for pair in intervals]
        else:
            tree.extend_filtration()
            pairs = [pair for dimension, pair in tree.extended_persistence(
                homology_coeff_field=11, min_persistence=MIN_PERSISTENCE)[1] if dimension == 2]
        return sorted(pairs, key=lambda pair: pair[0] - pair[1], reverse=True)

    def analyze(self, date, txt_files, pressure=None):
        """Return both full diagrams and score details for numerical validation."""
        values = self.get_values(date, txt_files, pressure=pressure)
        if self._templates is None:
            self._prepare_geometry()
        field = values.reshape(self.resolution)
        doubled = np.hstack((field, field)).ravel()
        bars = [self._pairs(template, data)
                for template, data in zip(self._templates, (values, doubled))]
        # Equal longest lifetimes can have different deaths. The application's
        # displacement uses the first such polar bar, so preserve GUDHI's order.
        if (self.method == 'planar' and len(bars[0]) > 1
                and bars[0][0][0] - bars[0][0][1] == bars[0][1][0] - bars[0][1][1]):
            if self._direct_templates is None:
                self._direct_templates = self._make_templates()
            bars = [self._gudhi_pairs(template, data, 'direct')
                    for template, data in zip(self._direct_templates, (values, doubled))]
        polar, grid = bars
        if not polar:
            return bars, {'empty_cyl': True, 'depths': [], 'split': None, 'disp': None, 'matched': None}
        depths = [birth - death for birth, death in polar]
        matched = next((pair for pair in grid if pair[1] == polar[0][1]), None)
        scores = {'depths': depths, 'split': depths[1] / depths[0] if len(depths) > 1 else 0.,
                  'disp': (matched[0] - matched[1]) / depths[0] if matched is not None else 0.,
                  'matched': matched}
        return bars, scores

    def compute_data(self, date, txt_files, pressure=None):
        """Return the application's [depths, split score, displacement score]."""
        _, scores = self.analyze(date, txt_files, pressure=pressure)
        if scores.get('empty_cyl'):
            raise ValueError('No positive-persistence polar feature for date ' + date)
        return [scores['depths'], scores['split'], scores['disp']]


def main():
    """Analyze dates on a server without importing the GUI or plotting stack."""
    import argparse
    import json
    import time
    from superpov_planar import NativeUnavailable

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', nargs='+', required=True, type=Path, help='Winter TSV files')
    parser.add_argument('--dates', nargs='+', required=True, help='Dates in YYYY-MM-DD format')
    parser.add_argument('--resolution', nargs=2, type=int, default=(46, 180),
                        metavar=('LATITUDES', 'LONGITUDES'))
    parser.add_argument('--pressure', type=float, help='Expected pressure in hPa, if specified')
    parser.add_argument('--method', choices=('auto', 'planar', 'direct', 'extended'), default='auto')
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument('--cache-dir', type=Path, default=Path('.superpov-cache'),
                               help='Persistent numeric cache (default: .superpov-cache)')
    cache_options.add_argument('--no-disk-cache', action='store_true', help='Use only in-memory caches')
    args = parser.parse_args()
    try:
        analyzer = DailyAnalyzer(args.resolution, method=args.method,
                                 cache_dir=None if args.no_disk_cache else args.cache_dir)
        for date in args.dates:
            started = time.perf_counter()
            depths, split, displacement = analyzer.compute_data(date, args.file, pressure=args.pressure)
            print(json.dumps({'date': date, 'depths': depths, 'split': split,
                              'displacement': displacement, 'method': analyzer.method,
                              'elapsed_seconds': time.perf_counter() - started}, allow_nan=False), flush=True)
    except (ValueError, OSError, NativeUnavailable) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
