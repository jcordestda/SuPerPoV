"""Optional, replaceable disk entries containing only JSON and numeric arrays.

Cache errors are misses: callers must be able to reconstruct every entry. Files
are uncompressed NPZ archives; neither Python objects nor native GUDHI objects
are serialized. A revision replaces the existing entry for the same kind/key.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np


SCHEMA_VERSION = 1
MAX_ENTRY_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
_METADATA = '__metadata__'
_NUMERIC_KINDS = frozenset('buifc')
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')


class DiskCache:
    """Best-effort numeric cache, creating its directory only when storing.

    ``metadata`` must be JSON serializable and describe every input needed to
    validate an entry. ``key`` identifies an entry independently of its current
    input revision. Loaded arrays are independent, read-only NumPy arrays.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, kind, key):
        if kind not in ('winter', 'geometry') or not isinstance(key, str):
            raise ValueError('Invalid cache kind or key')
        digest = hashlib.sha256((kind + '\0' + key).encode('utf-8')).hexdigest()
        return self.directory / f'{kind}-{digest}.npz'

    @staticmethod
    def _metadata_bytes(metadata):
        if not isinstance(metadata, dict):
            raise ValueError('Cache metadata must be a dictionary')
        envelope = {'schema_version': SCHEMA_VERSION, 'metadata': metadata}
        encoded = json.dumps(envelope, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')
        if len(encoded) > MAX_METADATA_BYTES:
            raise ValueError('Cache metadata is too large')
        return encoded

    @staticmethod
    def _read_array(archive, member):
        # Check the declared shape before np.load allocates memory. Even a tiny
        # corrupt NPY header can otherwise request an enormous allocation.
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError('Unsupported cache array format')
            if dtype.kind not in _NUMERIC_KINDS or any(size < 0 for size in shape):
                raise ValueError('Cache arrays must contain numeric data')
            if math.prod(shape) * dtype.itemsize != member.file_size - stream.tell():
                raise ValueError('Cache array size does not match its header')
            stream.seek(0)
            values = np.load(stream, allow_pickle=False)
        values.setflags(write=False)
        return values

    def load(self, kind, key, metadata):
        """Return matching arrays, or ``None`` if unavailable or invalid."""
        try:
            expected = self._metadata_bytes(metadata)
            with zipfile.ZipFile(self._path(kind, key)) as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if (not 1 <= len(members) <= 65 or len(set(names)) != len(names)
                        or sum(member.file_size for member in members) > MAX_ENTRY_BYTES
                        or any(member.compress_type != zipfile.ZIP_STORED for member in members)):
                    return None
                meta_member = archive.getinfo(_METADATA + '.npy')
                if meta_member.file_size > MAX_METADATA_BYTES + 1024:
                    return None
                encoded = self._read_array(archive, meta_member)
                if encoded.dtype != np.dtype('uint8') or encoded.ndim != 1:
                    return None
                # Canonical JSON distinguishes scalar types and validates the
                # document while permitting whitespace or key-order differences.
                actual = json.dumps(json.loads(encoded.tobytes()), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')
                if actual != expected:
                    return None
                result = {}
                for member in members:
                    if member.filename == _METADATA + '.npy':
                        continue
                    name = member.filename[:-4]
                    if not member.filename.endswith('.npy') or not _NAME.fullmatch(name):
                        return None
                    result[name] = self._read_array(archive, member)
                return result
        except (OSError, ValueError, TypeError, KeyError, EOFError, OverflowError,
                zipfile.BadZipFile, RuntimeError):
            return None

    def store(self, kind, key, metadata, arrays):
        """Atomically publish numeric arrays; return ``False`` on cache failure."""
        temporary = None
        try:
            destination = self._path(kind, key)
            encoded = self._metadata_bytes(metadata)
            payload = {}
            for name, value in arrays.items():
                if not isinstance(name, str) or not _NAME.fullmatch(name):
                    return False
                array = np.asarray(value)
                if array.dtype.kind not in _NUMERIC_KINDS:
                    return False
                payload[name] = array
            payload[_METADATA] = np.frombuffer(encoded, dtype=np.uint8)
            if (len(payload) > 65 or sum(array.nbytes + 1024 for array in payload.values())
                    > MAX_ENTRY_BYTES):
                return False
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w+b', prefix=destination.stem + '-',
                                             suffix='.tmp', dir=self.directory,
                                             delete=False) as stream:
                temporary = Path(stream.name)
                with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
                    for name, array in payload.items():
                        with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                            np.lib.format.write_array(member, array, version=(1, 0),
                                                      allow_pickle=False)
                stream.flush()
            os.replace(temporary, destination)
            return True
        except (OSError, ValueError, TypeError, AttributeError, OverflowError,
                zipfile.BadZipFile, RuntimeError):
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
