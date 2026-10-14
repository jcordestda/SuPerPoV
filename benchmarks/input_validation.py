"""Preflight guards for the ordered, single-pressure winter benchmark format."""

from datetime import datetime, timedelta
from pathlib import Path

def validate_output_path(destination, protected_paths):
    """Reject input/source aliases, directories, and Python output paths."""
    destination = Path(destination).resolve()
    if destination.is_dir() or destination.suffix.lower() in {'.py', '.pyw', '.pyc'}:
        raise ValueError('Output must be a result file, not a directory or Python file')
    for path in protected_paths:
        path = Path(path).resolve()
        if destination == path or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError('Output must not overwrite an input or source file')
    return destination


def validate_frame(frame, resolution, requested_dates=None, original_lookup=False):
    """Validate the existing row convention without reordering or changing data.

    The scripts assume complete north-to-south latitude rows and eastward
    longitude columns, with one daily block per date and one pressure level.
    This is deliberately not a general atmospheric-data loader.
    """
    import numpy as np

    required = {'# date', 'lev', 'lat', 'lon', 'data'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Expected nonempty TSV columns: # date, lev, lat, lon, data')
    if len(resolution) != 2 or min(resolution) < 2:
        raise ValueError('Resolution must contain two dimensions of at least 2')
    for column in ('lev', 'lat', 'lon', 'data'):
        try:
            finite = np.isfinite(frame[column].to_numpy(dtype=np.float64)).all()
        except (TypeError, ValueError) as error:
            raise ValueError(f'{column} must contain finite numeric values') from error
        if not finite:
            raise ValueError(f'{column} must contain finite numeric values')
    if frame['lev'].nunique() != 1:
        raise ValueError('Expected one pressure level in the input file')

    row_dates = frame['# date'].to_numpy()
    starts = np.r_[0, np.flatnonzero(row_dates[1:] != row_dates[:-1]) + 1]
    ends = np.r_[starts[1:], len(frame)]
    dates = [str(row_dates[index]) for index in starts]
    count = resolution[0] * resolution[1]
    if np.any(ends - starts != count):
        raise ValueError(f'Each consecutive date block must have {count} rows for resolution {tuple(resolution)}')
    parsed = []
    for date in dates:
        try:
            value = datetime.strptime(date, '%Y-%m-%d')
        except ValueError as error:
            raise ValueError(f'Invalid YYYY-MM-DD date: {date}') from error
        if value.strftime('%Y-%m-%d') != date or value.month not in (11, 12, 1, 2, 3, 4):
            raise ValueError('Dates must use YYYY-MM-DD within the November–April season')
        parsed.append(value)
    if any(right - left != timedelta(days=1) for left, right in zip(parsed, parsed[1:])):
        raise ValueError('Date blocks must be unique, chronological, and consecutive daily blocks')
    winter_years = {date.year if date.month >= 11 else date.year - 1 for date in parsed}
    if len(winter_years) != 1:
        raise ValueError('Expected dates from one November–April winter')

    expected_lat = np.repeat(np.linspace(90., 0., resolution[0]), resolution[1])
    expected_lon = np.tile(np.linspace(0., 360., resolution[1], endpoint=False), resolution[0])
    for column, expected in (('lat', expected_lat), ('lon', expected_lon)):
        actual = frame[column].to_numpy(dtype=np.float64).reshape(-1, count)
        if not np.allclose(actual, expected[None, :], rtol=0., atol=1e-10):
            raise ValueError(f'{column} grid must match resolution {tuple(resolution)} in north-to-south, eastward row order')

    if requested_dates is not None:
        if len(set(requested_dates)) != len(requested_dates):
            raise ValueError('Requested dates must be unique')
        missing = [date for date in requested_dates if date not in dates]
        if missing:
            raise ValueError(f'Requested dates are absent from the input: {", ".join(missing)}')
        if original_lookup:
            first_year = parsed[0].year
            for date in requested_dates:
                value = parsed[dates.index(date)]
                expected_year = value.year if value.month >= 11 else value.year - 1
                if first_year != expected_year:
                    raise ValueError('Original year lookup requires the file to begin in November or December of this winter')
                if date == dates[-1] and date != dates[0] and date[5:] != '04-30':
                    raise ValueError('Original date lookup cannot safely read this final date; include the following complete day, or use a winter ending April 30')
    return {'dates': dates, 'pressure_hpa': float(frame['lev'].iloc[0]),
            'rows_per_day': count, 'coordinate_atol': 1e-10}
