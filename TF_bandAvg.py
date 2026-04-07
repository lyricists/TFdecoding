# ============================================================
# Average frequency band response
# ============================================================

import numpy as np


def tf_bandAvg(data, freqs, bands=None):
    """
    Average TF response across predefined frequency bands
    and stack them along a new band axis.

    Parameters
    ----------
    data : ndarray
        Shape = (n_PCs, n_freqs, n_timepoints, n_subjects)

    bands : dict or None
        Example:
        {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 60),
        }

    Returns
    -------
    band_array : ndarray
        Shape = (n_components, n_bands, n_timepoints, n_subjects)

    band_names : list
        Order of bands in axis=1
    """

    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha_low": (8, 10),
            "alpha_high": (10, 13),
            "beta_low": (13, 15),
            "beta_mid": (15, 18),
            "beta_high": (18, 30),
            "gamma": (30, 60),
        }

    band_list = []
    band_names = []

    for band_name, (fmin, fmax) in bands.items():

        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        if not np.any(freq_mask):
            raise ValueError(
                f"No frequencies found for band {band_name}: {fmin}-{fmax} Hz"
            )

        band_avg = data[:, freq_mask, :, :].mean(axis=1)

        band_list.append(band_avg)
        band_names.append(band_name)

    band_array = np.stack(band_list, axis=1)

    return band_array, band_names
