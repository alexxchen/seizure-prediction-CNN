import mne
import numpy as np

from utils.load_signals import _pick_legacy_channels


def test_modern_duplicate_channel_names_preserve_legacy_electrodes_and_order():
    names = ['A', 'T8-P8-0', 'B', 'T8-P8-1']
    values = np.arange(4 * 10).reshape(4, 10).astype(float)
    raw = mne.io.RawArray(values, mne.create_info(names, sfreq=256, ch_types='eeg'), verbose=False)
    _pick_legacy_channels(raw, ['B', 'T8-P8', 'A'])
    assert raw.ch_names == ['B', 'T8-P8-0', 'A']
    np.testing.assert_array_equal(raw.get_data(), values[[2, 1, 0]])
