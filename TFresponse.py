# ============================================================
# Time-frequency analysis
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

    ch_names = [f"ch{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    epochs = mne.EpochsArray(data, info, tmin=-0.2, verbose=False)

    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        verbose=False,
        n_jobs=-1,
    )

    # Apply baseline z-score normalization
    tfr.apply_baseline(baseline=(-0.2, 0), mode="zscore")

    return tfr.data


# Parameters for TF decomposition
sfreq = 250

freqs = np.logspace(np.log10(4), np.log10(60), 50)  # Frequencies from 4 to 60 Hz
n_cycles = np.interp(
    np.log10(freqs), [np.log10(4), np.log10(60)], [2, 8]
)  # Varying cycles
# n_cycles = freqs / 4  # Number of cycles for each frequency

ch = [
    np.concatenate((np.array([3, 4]) - 1, np.array([1, 5]) + 31)),  # Left frontal
    np.concatenate(
        (np.array([1, 2, 32]) - 1, np.array([2, 3, 4, 30, 31]) + 31)
    ),  # Mid frontal
    np.concatenate((np.array([30, 31]) - 1, np.array([28, 29]) + 31)),  # Right frontal
    np.concatenate(
        (np.array([6, 8, 9, 11]) - 1, np.array([6, 7, 9, 10, 11]) + 31)
    ),  # Left central
    np.concatenate(
        (np.array([7, 12, 23, 24, 29]) - 1, np.array([8, 21, 25, 32]) + 31)
    ),  # Mid central
    np.concatenate(
        (np.array([22, 25, 26, 28]) - 1, np.array([22, 23, 24, 26, 27]) + 31)
    ),  # Right central
    np.concatenate((np.array([14, 15]) - 1, np.array([13, 14]) + 31)),  # Left parietal
    np.concatenate(
        (np.array([13, 16, 17, 18]) - 1, np.array([12, 15, 16, 17, 20]) + 31)
    ),  # Mid parietal
    np.concatenate((np.array([19, 20]) - 1, np.array([18, 19]) + 31)),  # Right parietal
]

# ROI averaging, time -> -200 - 1000 ms
data = np.empty((len(ch), 425, Dataset.shape[2], n_sub))

for i in range(len(ch)):
    data[i, :, :, :] = np.mean(Dataset[ch[i], :, :, :][:, np.arange(425), :, :], axis=0)

data = data.transpose(2, 0, 1, 3)  # n_trials, n_ROI, n_timepoints, n_subjects

conditions = ["Biography", "Action", "Reflection", "Intention", "All"]
types = ["positive", "negative"]

results = {
    cond: {t: np.empty((len(ch), len(freqs), 300, n_sub)) for t in types}
    for cond in conditions
}

print(f"Processing TF decomposition...")

for n in tqdm(range(n_sub)):

    tfdata = tf_decomposition(data[:, :, :, n], sfreq, freqs, n_cycles)

    idx_all = {t: np.array([], dtype=int) for t in types}

    for k in conditions:
        if k != "All":
            for t in types:
                idx = log[n][k][
                    t
                ]  # Get trial indices for the current condition and type
                results[k][t][:, :, :, n] = np.mean(
                    tfdata[idx, :, :, 0:300], axis=0
                )  # Average across trials for the current condition and type

                idx_all[t] = np.concatenate((idx_all[t], idx))

        else:
            for t in types:
                results[k][t][:, :, :, n] = np.mean(
                    tfdata[idx_all[t], :, :, 0:300], axis=0
                )  # Average across trials for the current condition and type


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open(
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/trf_response_full.pkl",
    "wb",
) as file:
    pickle.dump(results, file)
print("\nTF response analysis completed and saved.")
