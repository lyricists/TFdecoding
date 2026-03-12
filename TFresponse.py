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
# Parameters
# ------------------------------------------------------------
fPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/Modeling/Data/"
bPath = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/"
fileName = "train_test_PCA_BI_full.pkl"
TOI_list = ["Bio", "Int"]
n_sub = 137  # N subjects
n_fold = 5  # N PCA folds
fs = 250  # Sampling frequency (Hz)
results = {}

# ------------------------------------------------------------
# Loop over TOIs
# ------------------------------------------------------------
for TOI in TOI_list:
    print(f"\nProcessing TOI: {TOI}")

    with open(fPath + fileName, "rb") as file:
        Dataset = pickle.load(file)

    split_data = Dataset["split_data"][TOI]
    pcaDataset = Dataset["pcaData"]

    trf_response = []

    # --------------------------------------------------------
    # Subject Loop
    # --------------------------------------------------------
    for sub in tqdm(range(n_sub), desc=f"Subject Loop ({TOI})"):
        trf_sub = []

        for k in range(n_fold):
            EEG = np.concatenate(
                (
                    pcaDataset[k]["pcaData"][:, :, split_data[sub][k]["positive"], sub],
                    pcaDataset[k]["pcaData"][
                        :, :, split_data[sub][k]["test_positive"], sub
                    ],
                    pcaDataset[k]["pcaData"][:, :, split_data[sub][k]["negative"], sub],
                    pcaDataset[k]["pcaData"][
                        :, :, split_data[sub][k]["test_negative"], sub
                    ],
                ),
                axis=2,
            ).transpose(
                2, 0, 1
            )  # (trials, PCs, time points)

            info = mne.create_info(
                ch_names=[f"PC{i+1}" for i in range(EEG.shape[1])],
                sfreq=fs,
                ch_types="eeg",
            )

            epochs = mne.EpochsArray(EEG, info)

            freqs = np.linspace(1, 80, 80)  # Frequencies of interest
            n_cycles = freqs / 2.0  # Number of cycles for each frequency

            trf = epochs.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=False,
            )

            trf_sub.append(trf.data)

        trf.apply_baseline(mode="zscore", baseline=(0, 0.2))
        trf_response.append(np.mean(np.array(trf_sub), axis=0))

    results[TOI] = np.array(trf_response).transpose(0, 2, 3, 4, 1)[
        :, :, :, np.arange(300), :
    ]  # (subjects, PCs, frequencies, time points, epochs)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open(
    "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TF Analysis/Results/trf_response_full.pkl",
    "wb",
) as file:
    pickle.dump(results, file)
print("\nTF response analysis completed and saved.")
