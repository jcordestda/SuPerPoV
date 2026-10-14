"""Validate GUI inputs and construct calendar dates without starting Tk."""

from datetime import datetime, timedelta
from numbers import Integral
import re

import numpy as np
import pandas as pd


def _calendar_date(value):
    message = 'Please enter a valid date in YYYY-MM-DD format.'
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', value):
        raise ValueError(message)
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError as error:
        raise ValueError(message) from error


def _day_count(value, label):
    message = f'{label} must be a non-negative integer. Enter 0 for no extra days.'
    if isinstance(value, str):
        if not re.fullmatch(r'\+?[0-9]+', value.strip()):
            raise ValueError(message)
        try:
            value = int(value)
        except ValueError as error:
            raise ValueError(message) from error
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(message)
    return int(value)


def validate_date_window(value, before, after):
    """Return a valid winter datetime and non-negative integer day counts."""
    date = _calendar_date(value)
    if date.month not in (11, 12, 1, 2, 3, 4):
        raise ValueError('Please choose a November–April date for the winter data.')
    winter_year = date.year if date.month >= 11 else date.year - 1
    if not 1 <= winter_year < 9999:
        raise ValueError('Please choose a complete November–April winter within years 0001–9999.')
    return date, _day_count(before, 'Days before'), _day_count(after, 'Days after')


def winter_date_window(value, before, after):
    """Return chronological dates on either side, clipped to the same winter."""
    date, before, after = validate_date_window(value, before, after)
    winter_year = date.year if date.month >= 11 else date.year - 1
    first = datetime(winter_year, 11, 1)
    last = datetime(winter_year + 1, 4, 30)
    # Clip integers before allocating a range or constructing a timedelta.
    before = min(before, (date - first).days)
    after = min(after, (last - date).days)
    prior = [(date - timedelta(days=offset)).date().isoformat()
             for offset in range(before, 0, -1)]
    later = [(date + timedelta(days=offset)).date().isoformat()
             for offset in range(1, after + 1)]
    return prior, later


def read_wind_data(path):
    """Read the existing tab-separated DATES/SPEEDS format with clear errors."""
    try:
        frame = pd.read_csv(path, sep='\t', dtype={'DATES': str})
        if not {'DATES', 'SPEEDS'}.issubset(frame.columns) or frame.empty:
            raise ValueError('Expected DATES and SPEEDS columns with at least one row.')
        if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError('Wind data rows must have the same number of fields as the header.')
        for value in frame['DATES']:
            _calendar_date(value)
        if pd.api.types.is_bool_dtype(frame['SPEEDS']):
            raise ValueError('SPEEDS must contain numeric wind speeds; boolean values are invalid.')
        frame['SPEEDS'] = pd.to_numeric(frame['SPEEDS'], errors='raise')
        if not np.isfinite(frame['SPEEDS'].to_numpy()).all():
            raise ValueError('SPEEDS must contain finite numbers without missing values.')
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f'Cannot use wind speed file "{path}": {error}') from error
    return frame
