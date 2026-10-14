"""Persistent numeric cache behavior, including recovery from broken entries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

try:
    import superpov_cache as cache_module
except ModuleNotFoundError as error:
    if error.name != 'superpov_cache':
        raise
    cache_module = None


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cache_module, 'The persistent numeric cache is not implemented')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'cache'
        self.cache = cache_module.DiskCache(self.directory)
        self.metadata = {'fingerprint': [7, 19, 23], 'resolution': [4, 8]}
        self.arrays = {'values': np.array([[1., 2.], [3., 4.]]),
                       'dates': np.array([0, 1], dtype=np.int64)}

    def store(self, **overrides):
        arguments = dict(kind='winter', key='/source/winter.txt',
                         metadata=self.metadata, arrays=self.arrays)
        arguments.update(overrides)
        return self.cache.store(**arguments)

    def load(self, **overrides):
        arguments = dict(kind='winter', key='/source/winter.txt', metadata=self.metadata)
        arguments.update(overrides)
        return self.cache.load(**arguments)

    def entry_path(self):
        entries = list(self.directory.glob('*.npz'))
        self.assertEqual(len(entries), 1)
        return entries[0]

    def rewrite_entry(self, change):
        path = self.entry_path()
        with np.load(path, allow_pickle=False) as stored:
            contents = {name: stored[name] for name in stored.files}
        change(contents)
        np.savez(path, **contents)

    def test_missing_cache_does_not_create_directories(self):
        self.assertIsNone(self.load())
        self.assertFalse(self.directory.exists())

    def test_hit_survives_new_instance_and_returns_readonly_independent_arrays(self):
        self.assertTrue(self.store())
        self.arrays['values'][0, 0] = -99.
        self.cache = cache_module.DiskCache(self.directory)
        result = self.load()
        self.assertEqual(set(result), {'values', 'dates'})
        np.testing.assert_array_equal(result['values'], [[1., 2.], [3., 4.]])
        np.testing.assert_array_equal(result['dates'], [0, 1])
        self.assertEqual(result['dates'].dtype, np.dtype('int64'))
        self.assertFalse(result['values'].flags.writeable)
        result['values'].setflags(write=True)
        result['values'][0, 0] = 1000.
        self.assertEqual(self.load()['values'][0, 0], 1.)
        with zipfile.ZipFile(self.entry_path()) as archive:
            self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED
                                for item in archive.infolist()))

    def test_metadata_mismatch_misses_and_revision_overwrites_same_entry(self):
        self.assertTrue(self.store())
        changed = dict(self.metadata, fingerprint=[7, 19, 24])
        self.assertIsNone(self.load(metadata=changed))
        self.assertTrue(self.store(metadata=changed, arrays={'values': np.array([17.])}))
        self.assertIsNone(self.load())
        np.testing.assert_array_equal(self.load(metadata=changed)['values'], [17.])
        self.entry_path()

    def test_keys_and_kinds_are_isolated_and_cannot_escape_directory(self):
        for kind, key, value in [('winter', '../../outside', 1),
                                 ('geometry', '../../outside', 2),
                                 ('winter', 'different', 3)]:
            self.assertTrue(self.store(kind=kind, key=key, arrays={'values': np.array([value])}))
        for kind, key, value in [('winter', '../../outside', 1),
                                 ('geometry', '../../outside', 2),
                                 ('winter', 'different', 3)]:
            np.testing.assert_array_equal(self.load(kind=kind, key=key)['values'], [value])
        self.assertEqual(len(list(self.directory.glob('*.npz'))), 3)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [self.directory])

    def test_corrupt_and_truncated_entries_are_misses(self):
        self.assertTrue(self.store())
        path = self.entry_path()
        valid = path.read_bytes()
        for broken in [b'not an archive', valid[:len(valid) // 2], b'']:
            with self.subTest(length=len(broken)):
                path.write_bytes(broken)
                self.assertIsNone(self.load())

    def test_stale_schema_is_a_miss(self):
        self.assertTrue(self.store())

        def change(contents):
            envelope = json.loads(contents['__metadata__'].tobytes())
            envelope['schema_version'] = -1
            contents['__metadata__'] = np.frombuffer(json.dumps(envelope).encode(), dtype=np.uint8)

        self.rewrite_entry(change)
        self.assertIsNone(self.load())

    def test_malformed_metadata_is_a_miss(self):
        self.assertTrue(self.store())
        self.rewrite_entry(lambda arrays: arrays.update(__metadata__=np.array([255], dtype=np.uint8)))
        self.assertIsNone(self.load())

    def test_object_array_on_disk_is_rejected_without_pickle(self):
        self.assertTrue(self.store())
        self.rewrite_entry(lambda arrays: arrays.update(values=np.array([{'unexpected': 1}], dtype=object)))
        self.assertIsNone(self.load())

    def test_invalid_values_are_not_stored(self):
        for arrays in [{'values': np.array([object()], dtype=object)},
                       {'values': np.array(['a string'])},
                       {'../escape': np.array([1])},
                       {'__metadata__': np.array([1])}]:
            with self.subTest(arrays=list(arrays)):
                self.assertFalse(self.store(arrays=arrays))
        self.assertFalse(self.directory.exists())

    def test_invalid_metadata_is_not_stored(self):
        for metadata in [{'value': object()}, {'value': float('nan')}, {'value': float('inf')}]:
            with self.subTest(metadata=metadata):
                self.assertFalse(self.store(metadata=metadata))
        self.assertFalse(self.directory.exists())

    def test_oversized_entries_are_not_loaded_or_stored(self):
        self.assertTrue(self.store())
        with patch.object(cache_module, 'MAX_ENTRY_BYTES', 16):
            self.assertIsNone(self.load())
            self.assertFalse(self.store())

    def test_array_header_cannot_request_more_memory_than_its_member_size(self):
        self.assertTrue(self.store())
        path = self.entry_path()
        with zipfile.ZipFile(path) as archive:
            metadata = archive.read('__metadata__.npy')
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('__metadata__.npy', metadata)
            with archive.open('values.npy', 'w') as stream:
                np.lib.format.write_array_header_1_0(
                    stream, {'descr': '<f8', 'fortran_order': False, 'shape': (2 ** 40,)})
        self.assertIsNone(self.load())

    def test_failed_replacement_keeps_prior_entry_and_cleans_temporary_file(self):
        self.assertTrue(self.store())
        path = self.entry_path()
        previous = path.read_bytes()
        with patch.object(cache_module.os, 'replace', side_effect=OSError('storage unavailable')):
            self.assertFalse(self.store(arrays={'values': np.array([-1.])}))
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.directory.iterdir()), [path])
        np.testing.assert_array_equal(self.load()['values'], [[1., 2.], [3., 4.]])

    def test_interrupted_array_write_cleans_temporary_file(self):
        with patch.object(np.lib.format, 'write_array', side_effect=OSError('disk full')):
            self.assertFalse(self.store())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_unavailable_directory_is_a_miss_and_store_failure(self):
        self.directory.write_text('occupied by a regular file')
        self.assertIsNone(self.load())
        self.assertFalse(self.store())
        self.assertEqual(self.directory.read_text(), 'occupied by a regular file')

    def test_concurrent_writers_publish_one_complete_entry(self):
        def write(value):
            cache = cache_module.DiskCache(self.directory)
            return cache.store('geometry', 'same grid', {},
                               {'a': np.full(100, value), 'b': np.full(100, value)})

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertTrue(all(pool.map(write, range(12))))
        result = self.cache.load('geometry', 'same grid', {})
        np.testing.assert_array_equal(result['a'], result['b'])
        self.assertEqual(len(np.unique(result['a'])), 1)
        self.assertEqual(len(list(self.directory.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
