"""Cached daily SuPerPoV calculation, independent of the graphical interface.

One DailyAnalyzer shares validated winter arrays and fixed triangulations across
queries. It never caches completed scores or mutates a previously processed tree.
"""
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import operator
from pathlib import Path

import gudhi as gd
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


MIN_PERSISTENCE = 1e-5


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

    ``method='direct'`` computes finite ordinary H1 of negated heights.
    ``method='extended'`` retains extended persistence for numerical comparisons.
    Each instance is intended for one sequential analysis, not concurrent calls.
    """

    def __init__(self, resolution=(46, 180), method='direct', max_cached_files=8):
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
        if method not in ('direct', 'extended'):
            raise ValueError('Persistence method must be direct or extended')
        self.method = method
        self._headers = OrderedDict()
        self._files = OrderedDict()
        self._templates = None

    def _select_file(self, date, txt_files):
        year = _winter_year(_date(date))
        for filename in txt_files:
            path = Path(filename).resolve()
            signature = _fingerprint(path)
            cached = self._headers.get(path)
            if cached is None or cached[0] != signature:
                header = pd.read_csv(path, sep='\t', nrows=1)
                if header.empty or '# date' not in header:
                    raise ValueError('Winter files must have a nonempty # date column')
                file_year = _winter_year(_date(str(header['# date'].iloc[0])))
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

    def get_values(self, date, txt_files, pressure=None):
        """Return an independent float64 day array, loading its file only as needed."""
        path, signature, year = self._select_file(date, txt_files)
        cached = self._files.get(path)
        if cached is None or cached[0] != signature:
            winter = self._read_winter(path, year)
            if _fingerprint(path) != signature:
                raise ValueError('Input file changed while reading; retry the query')
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
        rows, columns = self.resolution
        r, theta = np.meshgrid(np.linspace(.001, 4., rows),
                               np.linspace(0., 2 * np.pi, columns, endpoint=False), indexing='ij')
        polar = np.column_stack(((r * np.cos(theta)).ravel(), (r * np.sin(theta)).ravel()))
        x, y = np.meshgrid(np.linspace(0., 90., rows),
                          np.linspace(0., 720., 2 * columns, endpoint=False), indexing='ij')
        grid = np.column_stack((x.ravel(), y.ravel()))
        templates = []
        for points in (polar, grid):
            triangles = Delaunay(points).simplices
            vertices = np.unique(triangles)[None, :]
            tree = gd.SimplexTree()
            tree.insert_batch(triangles.T, np.full(len(triangles), np.inf))
            # Batch insertion lowers vertices; repair raises higher simplices.
            # This pristine template is intentionally not a valid filtration.
            tree.reset_filtration(-np.inf, min_dim=1)
            templates.append((tree, vertices))
        self._templates = templates

    def _pairs(self, template, values):
        pristine, vertices = template
        tree = pristine.copy()
        filtration = -values if self.method == 'direct' else values
        tree.insert_batch(vertices, filtration[vertices[0]])
        tree.make_filtration_non_decreasing()
        if self.method == 'direct':
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', nargs='+', required=True, type=Path, help='Winter TSV files')
    parser.add_argument('--dates', nargs='+', required=True, help='Dates in YYYY-MM-DD format')
    parser.add_argument('--resolution', nargs=2, type=int, default=(46, 180),
                        metavar=('LATITUDES', 'LONGITUDES'))
    parser.add_argument('--pressure', type=float, help='Expected pressure in hPa, if specified')
    parser.add_argument('--method', choices=('direct', 'extended'), default='direct')
    args = parser.parse_args()
    try:
        analyzer = DailyAnalyzer(args.resolution, method=args.method)
        for date in args.dates:
            started = time.perf_counter()
            depths, split, displacement = analyzer.compute_data(date, args.file, pressure=args.pressure)
            print(json.dumps({'date': date, 'depths': depths, 'split': split,
                              'displacement': displacement,
                              'elapsed_seconds': time.perf_counter() - started}, allow_nan=False), flush=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
