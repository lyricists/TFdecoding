# ============================================================
# Time-frequency analysis
# Author: Woojae Jeong
# ============================================================

import numpy as np
import pickle
from tqdm import tqdm
import mne

mne.set_log_level("ERROR")

# ------------------------------------------------------------
# Data load
# ------------------------------------------------------------
fPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/"
bPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
fileName = "Data_sen_lepoch_full.pkl"  # Ch x Time x Trial x Subject
logName = "senIdx_TOI.pkl"
n_sub = 137  # N subjects

# EEG data load
with open(fPath + fileName, "rb") as file:
    Dataset = pickle.load(file)

# Log data load
with open(bPath + logName, "rb") as file:
    log = pickle.load(file)["Sentiment"]


# ------------------------------------------------------------
# TF Decomposition
# ------------------------------------------------------------


def tf_decomposition(data, sfreq, freqs, n_cycles):
    """
    Perform time-frequency decomposition using Morlet wavelets.

    Parameters:
    - data: 3D array (n_trials, n_channels, n_timepoints)
    - sfreq: Sampling frequency (Hz)
    - freqs: Array of frequencies to analyze (Hz)
    - n_cycles: Number of cycles in Morlet wavelet

    Returns:
    - power: 4D array (n_trials, n_channels, n_freqs, n_timepoints) of power values
    """

    n_trials, n_channels, n_timepoints = data.shape
    power = np.zeros((n_trials, n_channels, len(freqs), n_timepoints))

    for trial in tqdm(range(n_trials), desc="Processing trials"):
        for ch in range(n_channels):
            # Create MNE Epochs object for the current trial and channel
            info = mne.create_info(ch_names=[f"ch{ch}"], sfreq=sfreq)
            epochs = mne.EpochsArray(data[trial, ch][None, None, :], info)

            # Perform time-frequency decomposition
            tfr = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=False,
            )
            power[trial, ch] = tfr.data[0]

    return power


# Test test test


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

# with open(
#     "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TF Analysis/Results/trf_response_full.pkl",
#     "wb",
# ) as file:
#     pickle.dump(results, file)
# print("\nTF response analysis completed and saved.")
