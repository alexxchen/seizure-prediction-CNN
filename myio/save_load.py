import os
import pickle
import hickle
import h5py


def _load_legacy_array_cache(node):
    """Read the numeric list/array schema written by Python 2 hickle 2.1.0."""
    if isinstance(node, h5py.Dataset):
        if node.dtype.kind not in 'biufc':
            raise ValueError('Legacy cache contains unsupported non-numeric data')
        return node[()]
    keys = list(node.keys())
    if any(not key.startswith('data_') or not key[5:].isdigit() for key in keys):
        raise ValueError('Unsupported legacy cache structure')
    return [_load_legacy_array_cache(node[key])
            for key in sorted(keys, key=lambda key: int(key[5:]))]

def save_hickle_file(filename, data):
    filename = filename + '.hickle'
    print(('Saving to %s' % filename))

    temporary = filename + '.tmp'
    hickle.dump(data, temporary, mode='w')
    os.replace(temporary, filename)

def load_hickle_file(filename):
    filename = filename + '.hickle'
    if os.path.isfile(filename):
        print(('Loading %s ...' % filename))
        with h5py.File(filename, 'r') as handle:
            if handle.attrs.get('VERSION') == 3 and 'data_0' in handle:
                return _load_legacy_array_cache(handle['data_0'])
        data = hickle.load(filename)
        return data
    return None

def save_pickle_file(filename, data):
    filename = filename + '.pickle'
    print(('Saving to %s' % filename,))

    with open(filename, 'wb') as f:
        try:
            pickle.dump(data, f)
        except Exception:
            print(('Cannot pickle to %s' % filename))

def load_pickle_file(filename):
    filename = filename + '.pickle'
    if os.path.isfile(filename):
        print(('Loading %s ...' % filename,))
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            return data
    return None
