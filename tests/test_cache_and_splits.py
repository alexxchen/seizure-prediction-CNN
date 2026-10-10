import h5py
import numpy as np

from myio.save_load import load_hickle_file, save_hickle_file
from utils.prep_data import train_val_loo_split, train_val_test_split


def test_modern_hickle_round_trip(tmp_path):
    filename = str(tmp_path / 'features')
    arrays = [np.ones((3, 1, 6, 55, 114), dtype=np.float32),
              np.zeros((4, 1, 6, 55, 114), dtype=np.float32)]
    labels = [np.ones(3, dtype=np.int64), np.zeros(4, dtype=np.int64)]
    save_hickle_file(filename, [arrays, labels])
    restored = load_hickle_file(filename)
    for expected_group, actual_group in zip([arrays, labels], restored):
        for expected, actual in zip(expected_group, actual_group):
            np.testing.assert_array_equal(actual, expected)
    assert load_hickle_file(str(tmp_path / 'missing')) is None


def test_python2_legacy_cache_keeps_numeric_list_order(tmp_path):
    filename = str(tmp_path / 'legacy')
    with h5py.File(filename + '.hickle', 'w') as handle:
        handle.attrs['VERSION'] = 3
        root = handle.create_group('data_0')
        inputs = root.create_group('data_0')
        labels = root.create_group('data_1')
        for index in range(12):
            inputs.create_dataset(f'data_{index}', data=np.full((2, 1, 6, 55, 114), index))
            labels.create_dataset(f'data_{index}', data=np.ones(2, dtype=np.int64))
    restored_inputs, restored_labels = load_hickle_file(filename)
    assert len(restored_inputs) == len(restored_labels) == 12
    for index, inputs in enumerate(restored_inputs):
        assert np.all(inputs == index)


def make_split_data():
    labels = [np.array([1] * 8 + [2] * 8) for _ in range(4)]
    inputs = [np.full((16, 1, 1, 1, 1), index) for index in range(4)]
    negative_inputs = np.zeros((64, 1, 1, 1, 1))
    negative_labels = np.zeros(64, dtype=np.int64)
    return inputs, labels, negative_inputs, negative_labels


def test_leave_one_seizure_out_excludes_overlapped_test_windows():
    folds = list(train_val_loo_split(*make_split_data(), 0.25))
    assert len(folds) == 4
    for index, (train_X, train_y, val_X, val_y, test_X, test_y) in enumerate(folds):
        assert set(train_y) == set(val_y) == set(test_y) == {0, 1}
        assert (test_y == 1).sum() == 8
        assert np.all(test_X[test_y == 1] == index)
        assert index not in train_X[train_y == 1]


def test_temporal_split_holds_out_last_seizure():
    train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(
        *make_split_data(), 0.25, 0.35)
    assert np.all(test_X[test_y == 1] == 3)
    assert 3 not in train_X[train_y == 1]
    assert (test_y == 1).sum() == 8
    assert set(train_y) == set(val_y) == set(test_y) == {0, 1}
