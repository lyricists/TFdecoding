# ============================================================
# PCA-based Time-frequency analysis
# ============================================================

import numpy as np
import pickle
from tqdm import tqdm
import mne
from sklearn.decomposition import PCA
import mat73

mne.set_log_level("ERROR")


class TFDecompositionPCA:
    def __init__(
        self,
        fPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        save_path: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/",
        fileName: str = "Data_sen_lepoch_full.pkl",
        logName: str = "senIdx_TOI.pkl",
        chName: str = "GoodChannel.mat",
        subIdx: str = "subject_index.mat",
        n_sub: int = 137,
        n_components: int = 7,
        mode: str = None,
        tmin: int = -0.2,
        sfreq: int = 250,
    ):
        self.fPath = fPath
        self.bPath = bPath
        self.save_path = save_path
        self.fileName = fileName
        self.logName = logName
        self.chName = chName
        self.subIdx = subIdx
        self.n_sub = n_sub
        self.n_components = n_components
        self.mode = mode
        self.tmin = tmin
        self.sfreq = sfreq

        self.freqs = np.logspace(np.log10(4), np.log10(60), 50)
        self.n_cycles = np.interp(
            np.log10(self.freqs),
            [np.log10(4), np.log10(60)],
            [1, 10],
        )

        # Sentence categories and types
        self.conditions = ["Biography", "Action", "Reflection", "Intention", "All"]
        self.types = ["positive", "negative"]

    # ------------------------------------------------------------
    # TF decomposition
    # ------------------------------------------------------------
    def decomposition(self, data):
        """
        Perform time-frequency decomposition using Morlet wavelets.

        Parameters
        ----------
        data : ndarray
            Shape = (n_trials, n_components, n_timepoints)

        Returns
        -------
        power : ndarray
            Shape = (n_trials, n_components, n_freqs, n_timepoints)
        """

        if self.mode == "single":
            pass

        elif self.mode == "erp":
            data = data.mean(axis=0, keepdims=True)  # Average over trials for ERP mode

        elif self.mode == "residual":
            n_trials = data.shape[0]

            if n_trials < 2:
                raise ValueError(
                    "Leave-one-trial-out residual requires at least 2 trials."
                )

            # Sum across trials: components x time
            total = data.sum(axis=0, keepdims=True)

            # Leave-one-trial-out mean for each trial
            loo_mean = (total - data) / (n_trials - 1)

            # Residual = trial - mean(other trials)
            data = data - loo_mean

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        _, n_components, _ = data.shape

        ch_names = [f"pc{i+1}" for i in range(n_components)]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")

        epochs = mne.EpochsArray(data, info, tmin=self.tmin, verbose=False)

        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            return_itc=False,
            average=False,
            verbose=False,
            n_jobs=-1,
        )

        tfr.apply_baseline(baseline=(self.tmin, 0), mode="zscore")
        return tfr.data

    # ------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------
    def load_data(self):
        # Channel index load
        goodCh = (
            mat73.loadmat(self.bPath + self.chName)["Channel"].astype(int).ravel() - 1
        )

        # EEG data load: expected shape = channels x time x trials x subjects
        with open(self.fPath + self.fileName, "rb") as file:
            Dataset = pickle.load(file)

        Dataset = Dataset[goodCh, :, :, :]

        # Log data load
        with open(self.bPath + self.logName, "rb") as file:
            log = pickle.load(file)["Sentiment"]

        # Subject group load
        subject_group = mat73.loadmat(self.bPath + self.subIdx)["subject_index"].ravel()

        group_indices = {
            "Control": np.where(subject_group == 1)[0],
            "Depressed": np.where(subject_group == 2)[0],
            "Suicidal": np.where(subject_group == 3)[0],
        }

        return Dataset, log, group_indices

    # ------------------------------------------------------------
    # PCA spatial feature extraction
    # ------------------------------------------------------------
    def pca_extraction(self, Dataset, group_indices):
        """
        Fit PCA on group-averaged ERP maps, then project the full dataset.

        Parameters
        ----------
        Dataset : ndarray
            Shape = (n_channels, n_timepoints, n_trials, n_subjects)

        Returns
        -------
        Dataset_pca : ndarray
            Shape = (n_components, n_timepoints, n_trials, n_subjects)
        pca : PCA
            Fitted PCA object
        """
        group_inputs = {"Control": [], "Depressed": [], "Suicidal": []}

        for sub in range(self.n_sub):
            subject_mean = Dataset[:, :, :, sub].mean(axis=2)  # ch x time

            for group_name in group_inputs:
                if sub in group_indices[group_name]:
                    group_inputs[group_name].append(subject_mean)
                    break

        pca_input = np.concatenate(
            [
                np.mean(group_inputs[group_name], axis=0)
                for group_name in ["Control", "Depressed", "Suicidal"]
            ],
            axis=1,  # concatenate over time
        )  # shape: ch x (time * 3)

        pca = PCA(n_components=self.n_components)
        pca.fit(pca_input.T)  # samples x channels

        # Project full dataset
        # Original: ch x time x trials x subjects
        X = Dataset.reshape(Dataset.shape[0], -1).T  # (time*trials*subjects) x ch
        X_pca = pca.transform(X)  # samples x components

        Dataset_pca = X_pca.T.reshape(
            self.n_components,
            Dataset.shape[1],
            Dataset.shape[2],
            Dataset.shape[3],
        )  # comp x time x trials x subjects

        return Dataset_pca, pca

    # ------------------------------------------------------------
    # Result container
    # ------------------------------------------------------------
    def initialize_results(self, n_timepoints):
        results = {
            cond: {
                t: np.empty(
                    (self.n_components, len(self.freqs), n_timepoints, self.n_sub),
                    dtype=np.float32,
                )
                for t in self.types
            }
            for cond in self.conditions
        }
        return results

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------
    def run(self):
        Dataset, log, group_indices = self.load_data()

        # PCA projection
        Dataset_pca, pca = self.pca_extraction(Dataset, group_indices)

        # Reorder for TF: trials x components x time x subjects
        Dataset_pca = Dataset_pca.transpose(2, 0, 1, 3)

        n_keep_time = min(300, Dataset_pca.shape[2])
        results = self.initialize_results(n_keep_time)

        print("Processing TF decomposition...")

        for n in tqdm(range(self.n_sub)):
            idx_all = {t: np.array([], dtype=int) for t in self.types}

            for cond in self.conditions:
                if cond != "All":
                    for t in self.types:
                        idx = np.asarray(log[n][cond][t], dtype=int)

                        if len(idx) > 0:
                            subset = Dataset_pca[
                                idx, :, :, n
                            ]  # trials x components x time
                            tfdata = self.decomposition(
                                subset
                            )  # trials x components x freqs x time

                            results[cond][t][:, :, :, n] = np.mean(
                                tfdata[:, :, :, :n_keep_time], axis=0
                            )

                            idx_all[t] = np.concatenate((idx_all[t], idx))
                        else:
                            results[cond][t][:, :, :, n] = np.nan

                else:
                    for t in self.types:
                        if len(idx_all[t]) > 0:
                            subset = Dataset_pca[idx_all[t], :, :, n]
                            tfdata = self.decomposition(subset)

                            results[cond][t][:, :, :, n] = np.mean(
                                tfdata[:, :, :, :n_keep_time], axis=0
                            )
                        else:
                            results[cond][t][:, :, :, n] = np.nan

        # Save results
        with open(
            self.save_path + "trf_response_pca_" + self.mode + ".pkl", "wb"
        ) as file:
            pickle.dump(
                {
                    "results": results,
                    "pca_components": pca.components_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "freqs": self.freqs,
                    "n_cycles": self.n_cycles,
                    "sfreq": self.sfreq,
                    "tmin": self.tmin,
                },
                file,
            )

        print("\nTF response analysis completed and saved.")
        return results, pca


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    analyzer = TFDecompositionPCA(
        n_sub=137,
        n_components=7,
        mode="single",
    )
    results, pca = analyzer.run()
