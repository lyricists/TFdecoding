# ============================================================
# Time-frequency analysis (PCA)
# ============================================================

import numpy as np
import pickle
from tqdm import tqdm
import mne
from sklearn.decomposition import PCA
import mat73

mne.set_log_level("ERROR")


# ------------------------------------------------------------
# TF decomposition
# ------------------------------------------------------------


def tf_decomposition(data, sfreq, freqs, n_cycles):
    """
    Perform time-frequency decomposition using Morlet wavelets.

    Parameters:
    - data: 3D array (n_trials, n_PCs, n_timepoints)
    - sfreq: Sampling frequency (Hz)
    - freqs: Array of frequencies to analyze (Hz)
    - n_cycles: Number of cycles in Morlet wavelet

    Returns:
    - power: 4D array (n_trials, n_channels, n_freqs, n_timepoints) of power values
    """

    n_trials, n_PCs, n_timepoints = data.shape

    ch_names = [f"ch{i+1}" for i in range(n_PCs)]
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


# ------------------------------------------------------------
# Data load
# ------------------------------------------------------------
fPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/"
bPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
fileName = "Data_sen_lepoch_full.pkl"  # Ch x Time x Trial x Subject
logName = "senIdx_TOI.pkl"
chName = "GoodChannel.mat"
n_sub = 137  # N subjects

# Channel index load
goodCh = mat73.loadmat(bPath + chName)["Channel"].astype(int) - 1

# EEG data load
with open(fPath + fileName, "rb") as file:
    Dataset = pickle.load(file)

Dataset = Dataset[goodCh, :, :, :]

# Log data load
with open(bPath + logName, "rb") as file:
    log = pickle.load(file)["Sentiment"]

# Subject index load
subject_group = mat73.loadmat(
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/subject_index.mat"
)["subject_index"]

group_indices = {
    "Control": np.where(subject_group == 1)[0],
    "Depressed": np.where(subject_group == 2)[0],
    "Suicidal": np.where(subject_group == 3)[0],
}

# ------------------------------------------------------------
# PCA spatial feature extraction
# ------------------------------------------------------------

group_inputs = {"Control": [], "Depressed": [], "Suicidal": []}

for sub in range(n_sub):
    subject_mean = Dataset[:, :, :, sub].mean(axis=2)

    for group in group_inputs:
        if sub in group_indices[group]:
            group_inputs[group].append(subject_mean)
            break

pca_input = np.concatenate(
    [
        np.mean(group_inputs[group], axis=0)
        for group in ["Control", "Depressed", "Suicidal"]
    ],
    axis=1,
)

# PCA fitting
pca = PCA(n_components=7)  # Number of PCs to retain (adjust as needed)
pca.fit(pca_input.T)

# Transform the original data using the fitted PCA

x_pca = pca.fit_transform(Dataset.reshape(Dataset.shape[0], -1).T)

Dataset = x_pca.T.reshape(len(pca.components_), *Dataset.shape[1:])

# Parameters for TF decomposition
sfreq = 250

freqs = np.logspace(np.log10(4), np.log10(60), 50)  # Frequencies from 4 to 60 Hz
n_cycles = np.interp(
    np.log10(freqs), [np.log10(4), np.log10(60)], [1, 10]
)  # Varying cycles

Dataset = Dataset.transpose(2, 0, 1, 3)  # n_trials, n_PCs, n_timepoints, n_subjects

conditions = ["Biography", "Action", "Reflection", "Intention", "All"]
types = ["positive", "negative"]

results = {
    cond: {t: np.empty((len(pca.components_), len(freqs), 300, n_sub)) for t in types}
    for cond in conditions
}

print(f"Processing TF decomposition...")

for n in tqdm(range(n_sub)):

    tfdata = tf_decomposition(Dataset[:, :, :, n], sfreq, freqs, n_cycles)

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
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/trf_response_pca.pkl",
    "wb",
) as file:
    pickle.dump(results, file)
print("\nTF response analysis completed and saved.")
