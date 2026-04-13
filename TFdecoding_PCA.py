# ============================================================
# PCA-based Time-frequency band decoding
# Modes:
#   - "evoked"  : bootstrap-average in time domain first, then TF
#   - "induced" : TF on single pseudo-trials first, baseline, then
#                 average power across sampled pseudo-trials
# ============================================================

import os
import pickle
import gc
import numpy as np
import mat73
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import mne

mne.set_log_level("ERROR")


class TFBandDecoder:
    def __init__(
        self,
        fpath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        save_dir: str = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/",
        fileName: str = "Data_sen_lepoch_full.pkl",
        IdxName: str = "subject_index.mat",
        logName: str = "senIdx_congruent.pkl",
        chName: str = "GoodChannel.mat",
        k_fold: int = 5,
        numPC: int = 3,
        Trial_num: int = 250,
        avg_num: int = 12,
        sfreq: int = 250,
        tmin: float = -0.2,
        tmax: float = 1.5,  # full epoch for TF
        decode_tmin: float = -0.2,  # decoding window start
        decode_tmax: float = 1.0,  # decoding window end
        fmin: float = 4.0,
        fmax: float = 60.0,
        n_freqs: int = 50,
        mode: str = "evoked",  # "evoked" or "induced"
        state: int = 42,
        saveName: str = "TFBandDecoding.pkl",
        dtype=np.float32,
        bands: dict = None,
    ):
        self.fpath = fpath
        self.bPath = bPath
        self.save_dir = save_dir
        self.fileName = fileName
        self.IdxName = IdxName
        self.logName = logName
        self.chName = chName

        self.kfold = k_fold
        self.numPC = numPC
        self.Trial_num = Trial_num
        self.avg_num = avg_num
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.decode_tmin = decode_tmin
        self.decode_tmax = decode_tmax
        self.mode = mode.lower()
        self.state = state
        self.saveName = saveName
        self.dtype = dtype

        if self.mode not in {"evoked", "induced"}:
            raise ValueError("mode must be either 'evoked' or 'induced'")

        self.freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs).astype(
            np.float32
        )
        self.n_cycles = np.interp(
            np.log10(self.freqs),
            [np.log10(fmin), np.log10(fmax)],
            [1, 10],
        ).astype(np.float32)

        if bands is None:
            self.bands = {
                "theta": (4, 8),
                "alpha_low": (8, 10),
                "alpha_high": (10, 13),
                "beta_low": (13, 15),
                "beta_mid": (15, 18),
                "beta_high": (18, 30),
                "gamma": (30, 60),
            }
        else:
            self.bands = bands

        self.band_names = list(self.bands.keys())
        self.rng = np.random.default_rng(self.state)

        self.run()

    # ------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------
    def load_EEG(self):
        print("Loading dataset")

        with open(os.path.join(self.fpath, self.fileName), "rb") as file:
            self.Dataset = pickle.load(file)

        self.subIdx = mat73.loadmat(os.path.join(self.bPath, self.IdxName))[
            "subject_index"
        ].ravel()

        self.goodCh = (
            mat73.loadmat(os.path.join(self.bPath, self.chName))["Channel"]
            .astype(int)
            .ravel()
            - 1
        )

        with open(os.path.join(self.bPath, self.logName), "rb") as file:
            self.senId = pickle.load(file)["Sentiment"]

        # shape: ch x time x trials x subjects
        self.Dataset = self.Dataset[self.goodCh, :, :, :].astype(self.dtype, copy=False)

        self.n_channels, self.n_times, self.n_trials, self.n_sub = self.Dataset.shape
        self.times = (
            np.arange(self.n_times, dtype=np.float32) / self.sfreq + self.tmin
        ).astype(np.float32)

        expected_n_times_1 = int(round((self.tmax - self.tmin) * self.sfreq)) + 1
        expected_n_times_2 = int(round((self.tmax - self.tmin) * self.sfreq))

        if self.n_times not in (expected_n_times_1, expected_n_times_2):
            print(
                f"Warning: n_times={self.n_times}, while expected around "
                f"{expected_n_times_2} or {expected_n_times_1} for "
                f"{self.tmin} to {self.tmax} s at {self.sfreq} Hz."
            )

        self.decode_mask = (self.times > self.decode_tmin) & (
            self.times <= self.decode_tmax
        )
        self.decode_times = self.times[self.decode_mask]

        if self.decode_times.size == 0:
            raise ValueError("Decoding window is empty. Check decode_tmin/decode_tmax.")

        print(
            f"TF epoch: {self.tmin:.3f} to {self.times[-1]:.3f} s "
            f"({self.n_times} samples)"
        )
        print(
            f"Decode window: {self.decode_tmin:.3f} to {self.decode_tmax:.3f} s "
            f"({self.decode_times.size} samples)"
        )
        print(f"Mode: {self.mode}")

    # ------------------------------------------------------------
    # Train/test split per subject, separately for pos and neg
    # ------------------------------------------------------------
    def train_test_split(self):
        print("Performing train/test split")
        skf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.state)

        self.split_data = []

        for n in tqdm(range(self.n_sub)):
            posIdx = np.asarray(self.senId[n]["positive"], dtype=int)
            negIdx = np.asarray(self.senId[n]["negative"], dtype=int)

            if len(posIdx) < self.kfold or len(negIdx) < self.kfold:
                raise ValueError(
                    f"Subject {n}: not enough trials for {self.kfold}-fold split "
                    f"(positive={len(posIdx)}, negative={len(negIdx)})"
                )

            p_folds, n_folds = [], []

            for train_idx, test_idx in skf.split(posIdx):
                p_folds.append(
                    {
                        "train": posIdx[train_idx],
                        "test": posIdx[test_idx],
                    }
                )

            for train_idx, test_idx in skf.split(negIdx):
                n_folds.append(
                    {
                        "train": negIdx[train_idx],
                        "test": negIdx[test_idx],
                    }
                )

            self.split_data.append(
                {
                    "positive": p_folds,
                    "negative": n_folds,
                }
            )

    # ------------------------------------------------------------
    # Fit fold-specific PCA using only training data
    # ------------------------------------------------------------
    def fit_pca_for_fold(self, k):
        cPos, cNeg = [], []
        dPos, dNeg = [], []
        sPos, sNeg = [], []

        for n in range(self.n_sub):
            pos_train = self.split_data[n]["positive"][k]["train"]
            neg_train = self.split_data[n]["negative"][k]["train"]

            subj_data = self.Dataset[:, :, :, n]  # ch x time x trial

            pos_mean = np.mean(subj_data[:, :, pos_train], axis=2)
            neg_mean = np.mean(subj_data[:, :, neg_train], axis=2)

            if self.subIdx[n] == 1:
                cPos.append(pos_mean)
                cNeg.append(neg_mean)
            elif self.subIdx[n] == 2:
                dPos.append(pos_mean)
                dNeg.append(neg_mean)
            else:
                sPos.append(pos_mean)
                sNeg.append(neg_mean)

        pca_input = np.concatenate(
            (
                (
                    np.mean(np.stack(cPos, axis=0), axis=0)
                    + np.mean(np.stack(cNeg, axis=0), axis=0)
                )
                / 2,
                (
                    np.mean(np.stack(dPos, axis=0), axis=0)
                    + np.mean(np.stack(dNeg, axis=0), axis=0)
                )
                / 2,
                (
                    np.mean(np.stack(sPos, axis=0), axis=0)
                    + np.mean(np.stack(sNeg, axis=0), axis=0)
                )
                / 2,
            ),
            axis=1,
        )  # ch x (time*3)

        pca = PCA(n_components=self.numPC, random_state=self.state)
        pca.fit(pca_input.T)

        return pca

    # ------------------------------------------------------------
    # Project one subject to PCA space
    # Input: ch x time x trial
    # Output: comp x time x trial
    # ------------------------------------------------------------
    def project_subject_pca(self, subj_data, pca):
        n_ch, n_time, n_trial = subj_data.shape
        X = subj_data.reshape(n_ch, -1).T  # (time*trial) x ch
        Xp = pca.transform(X).astype(self.dtype, copy=False)
        Xp = Xp.T.reshape(self.numPC, n_time, n_trial)
        return Xp

    # ------------------------------------------------------------
    # Bootstrap average in time domain (for evoked)
    # Input: comp x time x trials
    # Output: n_aug x comp x time
    # ------------------------------------------------------------
    def bootstrap_average_time(self, data, idx_pool, n_aug, rng):
        idx_pool = np.asarray(idx_pool, dtype=int)

        if len(idx_pool) == 0:
            raise ValueError("Empty idx_pool in bootstrap_average_time")

        sampled = rng.choice(idx_pool, size=(n_aug, self.avg_num), replace=True)
        sampled_data = data[:, :, sampled]  # comp x time x n_aug x avg_num
        pseudo = sampled_data.mean(axis=-1)  # comp x time x n_aug
        pseudo = np.transpose(pseudo, (2, 0, 1))  # n_aug x comp x time

        return pseudo.astype(self.dtype, copy=False)

    # ------------------------------------------------------------
    # Build bootstrap index matrix only (for induced)
    # Output: n_aug x avg_num
    # ------------------------------------------------------------
    def bootstrap_sample_indices(self, idx_pool, n_aug, rng):
        idx_pool = np.asarray(idx_pool, dtype=int)

        if len(idx_pool) == 0:
            raise ValueError("Empty idx_pool in bootstrap_sample_indices")

        sampled = rng.choice(idx_pool, size=(n_aug, self.avg_num), replace=True)
        return sampled

    # ------------------------------------------------------------
    # TF decomposition
    # Input: epochs x comp x time
    # Output: epochs x comp x freq x time
    # ------------------------------------------------------------
    def tf_decompose(self, data):
        if not np.all(np.isfinite(data)):
            raise ValueError("Input to TF decomposition contains NaN or Inf.")

        power = mne.time_frequency.tfr_array_morlet(
            data,
            sfreq=self.sfreq,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            output="power",
            zero_mean=True,
            n_jobs=1,
            verbose=False,
        ).astype(self.dtype, copy=False)

        if not np.all(np.isfinite(power)):
            raise ValueError("Raw TF power contains NaN or Inf.")

        # Allow tiny numerical negatives only
        if np.nanmin(power) < -1e-7:
            raise ValueError("Raw TF power contains strongly negative values.")

        power = np.maximum(power, 0.0).astype(self.dtype, copy=False)

        baseline_mask = (self.times >= self.tmin) & (self.times <= 0)
        if not np.any(baseline_mask):
            raise ValueError("Baseline window is empty")

        baseline = power[..., baseline_mask]  # epochs x comp x freq x baseline_time
        b_mean = baseline.mean(axis=-1, keepdims=True)
        b_std = baseline.std(axis=-1, keepdims=True)
        b_std[b_std < 1e-12] = 1.0

        power = (power - b_mean) / b_std

        if not np.all(np.isfinite(power)):
            raise ValueError("Baseline-normalized TF power contains NaN or Inf.")

        return power.astype(self.dtype, copy=False)

    # ------------------------------------------------------------
    # Average TF over predefined frequency bands
    # Input: epochs x comp x freq x time
    # Output: epochs x comp x band x time
    # ------------------------------------------------------------
    def tf_band_average(self, tf_data):
        band_data = []

        for band_name, (fmin, fmax) in self.bands.items():
            mask = (self.freqs >= fmin) & (self.freqs <= fmax)
            if not np.any(mask):
                raise ValueError(
                    f"No frequencies found in band {band_name}: {fmin}-{fmax} Hz"
                )

            band_avg = tf_data[:, :, mask, :].mean(axis=2)  # epochs x comp x time
            band_data.append(band_avg)

        band_data = np.stack(band_data, axis=2)  # epochs x comp x band x time
        return band_data.astype(self.dtype, copy=False)

    # ------------------------------------------------------------
    # Feature generation for evoked
    # Output: epochs x comp x band x time
    # ------------------------------------------------------------
    def make_features_evoked(self, subj_pca, idx_pool, n_aug):
        pseudo = self.bootstrap_average_time(subj_pca, idx_pool, n_aug, self.rng)
        tf = self.tf_decompose(pseudo)
        band = self.tf_band_average(tf)
        return band[:, :, :, self.decode_mask]

    # ------------------------------------------------------------
    # Feature generation for induced
    # TF each sampled trial first, then average power within each
    # bootstrap set
    # Output: epochs x comp x band x time
    # ------------------------------------------------------------
    def make_features_induced(self, subj_pca, idx_pool, n_aug):
        sampled = self.bootstrap_sample_indices(idx_pool, n_aug, self.rng)
        unique_trials, inverse = np.unique(sampled, return_inverse=True)

        # TF only once per unique trial
        single_trials = np.transpose(
            subj_pca[:, :, unique_trials], (2, 0, 1)
        )  # n_unique x comp x time

        tf_unique = self.tf_decompose(single_trials)  # n_unique x comp x freq x time
        band_unique = self.tf_band_average(tf_unique)  # n_unique x comp x band x time

        # Reconstruct sampled bootstrap sets and average power
        band_sampled = band_unique[inverse].reshape(
            n_aug,
            self.avg_num,
            self.numPC,
            len(self.band_names),
            self.n_times,
        )  # n_aug x avg_num x comp x band x time

        band_avg = band_sampled.mean(axis=1)  # n_aug x comp x band x time
        return band_avg[:, :, :, self.decode_mask].astype(self.dtype, copy=False)

    # ------------------------------------------------------------
    # Generate class labels
    # ------------------------------------------------------------
    def gen_class_labels(self):
        n_train_each = int(self.Trial_num * 0.8)
        n_test_each = int(self.Trial_num * 0.2)

        y_train = np.concatenate(
            [
                np.zeros(n_train_each, dtype=int),
                np.ones(n_train_each, dtype=int),
            ]
        )
        y_test = np.concatenate(
            [
                np.zeros(n_test_each, dtype=int),
                np.ones(n_test_each, dtype=int),
            ]
        )

        return y_train, y_test

    # ------------------------------------------------------------
    # Decode one subject, one fold
    # band_train/test shape: epochs x comp x band x time
    # Return: band x time
    # ------------------------------------------------------------
    def decode_band_timecourse(self, band_train, band_test):
        y_train, y_test = self.gen_class_labels()
        n_bands = band_train.shape[2]
        n_times = band_train.shape[3]

        scores = np.zeros((n_bands, n_times), dtype=self.dtype)

        for b in range(n_bands):
            for t in range(n_times):
                x_train = band_train[:, :, b, t]  # epochs x comp
                x_test = band_test[:, :, b, t]

                train_perm = self.rng.permutation(len(y_train))
                test_perm = self.rng.permutation(len(y_test))

                x_train = x_train[train_perm]
                x_test = x_test[test_perm]
                yt = y_train[train_perm]
                yv = y_test[test_perm]

                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

                clf = SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight="balanced",
                    tol=1e-3,
                    random_state=self.state,
                )
                clf.fit(x_train, yt)
                pred = clf.predict(x_test)
                scores[b, t] = balanced_accuracy_score(yv, pred)

        return scores

    # ------------------------------------------------------------
    # Save decoding results only
    # ------------------------------------------------------------
    def save_data(self):
        os.makedirs(self.save_dir, exist_ok=True)

        out = {
            "decode": self.decode_scores,  # subjects x bands x decode_time
            "decode_folds": self.decode_scores_folds,  # subjects x folds x bands x decode_time
            "band_names": self.band_names,
            "times_decode": self.decode_times,
            "freqs": self.freqs,
            "sfreq": self.sfreq,
            "mode": self.mode,
            "numPC": self.numPC,
            "subIdx": self.subIdx,
            "decode_tmin": self.decode_tmin,
            "decode_tmax": self.decode_tmax,
            "avg_num": self.avg_num,
            "Trial_num": self.Trial_num,
        }

        save_path = os.path.join(self.save_dir, self.saveName)
        with open(save_path, "wb") as file:
            pickle.dump(out, file)

        print(f"Saved: {save_path}")

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------
    def run(self):
        self.load_EEG()
        self.train_test_split()

        n_bands = len(self.band_names)
        self.decode_scores_folds = np.zeros(
            (self.n_sub, self.kfold, n_bands, self.decode_times.size),
            dtype=self.dtype,
        )

        print("Running fold-specific PCA + TF-band decoding")

        for k in range(self.kfold):
            print(f"\nFold {k+1}/{self.kfold}")

            pca = self.fit_pca_for_fold(k)

            for n in tqdm(range(self.n_sub), desc=f"Subjects fold {k+1}"):
                subj_raw = self.Dataset[:, :, :, n]  # ch x time x trial
                subj_pca = self.project_subject_pca(
                    subj_raw, pca
                )  # comp x time x trial

                pos_train = self.split_data[n]["positive"][k]["train"]
                pos_test = self.split_data[n]["positive"][k]["test"]
                neg_train = self.split_data[n]["negative"][k]["train"]
                neg_test = self.split_data[n]["negative"][k]["test"]

                n_train_each = int(self.Trial_num * 0.8)
                n_test_each = int(self.Trial_num * 0.2)

                if self.mode == "evoked":
                    pos_train_feat = self.make_features_evoked(
                        subj_pca, pos_train, n_train_each
                    )
                    neg_train_feat = self.make_features_evoked(
                        subj_pca, neg_train, n_train_each
                    )
                    pos_test_feat = self.make_features_evoked(
                        subj_pca, pos_test, n_test_each
                    )
                    neg_test_feat = self.make_features_evoked(
                        subj_pca, neg_test, n_test_each
                    )

                elif self.mode == "induced":
                    pos_train_feat = self.make_features_induced(
                        subj_pca, pos_train, n_train_each
                    )
                    neg_train_feat = self.make_features_induced(
                        subj_pca, neg_train, n_train_each
                    )
                    pos_test_feat = self.make_features_induced(
                        subj_pca, pos_test, n_test_each
                    )
                    neg_test_feat = self.make_features_induced(
                        subj_pca, neg_test, n_test_each
                    )

                train_feat = np.concatenate([pos_train_feat, neg_train_feat], axis=0)
                test_feat = np.concatenate([pos_test_feat, neg_test_feat], axis=0)

                scores = self.decode_band_timecourse(train_feat, test_feat)
                self.decode_scores_folds[n, k] = scores

                del subj_raw, subj_pca
                del pos_train_feat, neg_train_feat, pos_test_feat, neg_test_feat
                del train_feat, test_feat, scores
                gc.collect()

            del pca
            gc.collect()

        self.decode_scores = np.mean(self.decode_scores_folds, axis=1)

        print("\nSaving results")
        self.save_data()


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":

    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60),
    }

    decoder = TFBandDecoder(
        numPC=3,
        k_fold=5,
        Trial_num=250,
        avg_num=12,
        tmin=-0.2,
        tmax=1.5,  # TF runs on full -200 to 1500 ms
        decode_tmin=-0.2,
        decode_tmax=1.0,  # decoding runs on -200 to 1000 ms
        mode="evoked",  # "evoked" or "induced"
        bands=bands,
        saveName="TFBandDecoding_3pc_evoked.pkl",
    )
