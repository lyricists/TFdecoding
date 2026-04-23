# ============================================================
# PCA-based TF time-frequency band class decoding
# ------------------------------------------------------------
# Group comparisons:
#   control_vs_depressed          : 1 vs 2
#   depressed_vs_suicidal         : 2 vs 3
#   control_vs_suicidal           : 1 vs 3
#   control_vs_depressedsuicidal  : 1 vs (2 or 3)
#
# For each fold:
#   1) split subjects
#   2) fit PCA on ALL training trials from training subjects only
#   3) project train/test subjects into fold-specific PC space
# Supports:
#   - induced : TF on single trials, then average TF across trials
#   - evoked  : average trials first, then TF on averaged signal
#   4) run TF decomposition on PC signals
#   5) baseline correct
#   6) average within frequency bands
#   7) build subject-level maps for each category/contrast
#   8) decode at each time point with train-only feature scaling
#
# Categories:
#   Biography, Action, Reflection, Intention, All
#
# Contrasts:
#   positive, negative, neg_minus_pos
#
# Classifiers:
#   - "svm"    -> LinearSVC
#   - "logreg" -> LogisticRegression
#
# Output:
#   results[comparison][category][contrast]["score"]
# ============================================================

import os
import pickle
import gc
import numpy as np
import mat73
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import mne

mne.set_log_level("ERROR")


class TFBandDecoder:
    def __init__(
        self,
        fpath="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Preprocessed data/",
        bPath="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Data/Behavior/",
        save_dir="/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/",
        fileName="Data_sen_lepoch_full_long.pkl",
        IdxName="subject_index.mat",
        logName="senIdx_TOI.pkl",
        chName="GoodChannel.mat",
        k_fold=5,
        numPC=3,
        sfreq=250,
        tmin=-0.3,
        tmax=1.5,
        decode_tmin=-0.2,
        decode_tmax=1.0,
        fmin=4.0,
        fmax=60.0,
        n_freqs=50,
        baseline=(-0.2, 0.0),
        baseline_mode="zscore",  # "logratio", "ratio", "percent", "zscore", "mean"
        response_mode="induced",  # "induced" or "evoked"
        state=42,
        saveName="TFBandDecoding_shared_foldPCA.pkl",
        dtype=np.float32,
        bands=None,
        categories=None,
        comparisons=None,
        classifier="svm",  # "svm" or "logreg"
        clf_c=1.0,
        raw_pca_standardize=False,
        pca_trial_cap_per_subject=None,
        tf_n_jobs=1,
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
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.decode_tmin = decode_tmin
        self.decode_tmax = decode_tmax
        self.baseline = baseline
        self.baseline_mode = baseline_mode
        self.response_mode = response_mode.lower()
        self.state = state
        self.saveName = saveName
        self.dtype = dtype
        self.raw_pca_standardize = raw_pca_standardize
        self.pca_trial_cap_per_subject = pca_trial_cap_per_subject
        self.tf_n_jobs = tf_n_jobs

        if self.response_mode not in {"induced", "evoked"}:
            raise ValueError("response_mode must be either 'induced' or 'evoked'")

        self.classifier = classifier.lower()
        self.clf_c = clf_c
        if self.classifier not in {"svm", "logreg"}:
            raise ValueError("classifier must be either 'svm' or 'logreg'")

        self.freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs).astype(
            self.dtype
        )
        self.n_cycles = np.interp(
            np.log10(self.freqs),
            [np.log10(fmin), np.log10(fmax)],
            [1.0, 10.0],
        ).astype(self.dtype)

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

        if categories is None:
            self.categories = ["Biography", "Action", "Reflection", "Intention", "All"]
        else:
            self.categories = categories

        if comparisons is None:
            self.comparisons = {
                "control_vs_depressed": {"include": [1, 2], "map": {1: 0, 2: 1}},
                "depressed_vs_suicidal": {"include": [2, 3], "map": {2: 0, 3: 1}},
                "control_vs_suicidal": {"include": [1, 3], "map": {1: 0, 3: 1}},
                "control_vs_depressedsuicidal": {
                    "include": [1, 2, 3],
                    "map": {1: 0, 2: 1, 3: 1},
                },
            }
        else:
            self.comparisons = comparisons

        self.contrasts = ["positive", "negative", "neg_minus_pos"]
        self.band_names = list(self.bands.keys())
        self.rng = np.random.default_rng(self.state)

        self.run()

    # ------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------
    def load_EEG(self):
        print("Loading dataset...")

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

        self.Dataset = self.Dataset[self.goodCh, :, :, :].astype(self.dtype, copy=False)

        self.n_channels, self.n_times, self.n_trials, self.n_sub = self.Dataset.shape
        self.times = (
            np.arange(self.n_times, dtype=self.dtype) / self.sfreq + self.tmin
        ).astype(self.dtype)

        expected_n_times_1 = int(round((self.tmax - self.tmin) * self.sfreq)) + 1
        expected_n_times_2 = int(round((self.tmax - self.tmin) * self.sfreq))

        if self.n_times not in (expected_n_times_1, expected_n_times_2):
            print(
                f"Warning: n_times={self.n_times}, expected around "
                f"{expected_n_times_2} or {expected_n_times_1} for "
                f"{self.tmin} to {self.tmax}s at {self.sfreq}Hz."
            )

        self.decode_mask = (self.times >= self.decode_tmin) & (
            self.times <= self.decode_tmax
        )
        self.decode_times = self.times[self.decode_mask]

        if self.decode_times.size == 0:
            raise ValueError("Decode window is empty.")

        self.base_mask = (self.times >= self.baseline[0]) & (
            self.times <= self.baseline[1]
        )
        if not np.any(self.base_mask):
            raise ValueError("Baseline window is empty.")

        if len(self.senId) != self.n_sub:
            raise ValueError(
                f"Mismatch: len(senId)={len(self.senId)} but n_sub={self.n_sub}"
            )

        print(
            f"Raw epoch: {self.tmin:.3f} to {self.times[-1]:.3f} s "
            f"({self.n_times} samples)"
        )
        print(
            f"Decode window: {self.decode_tmin:.3f} to {self.decode_tmax:.3f} s "
            f"({self.decode_times.size} samples)"
        )
        print(
            f"Shared fold-specific PCA before TF | "
            f"classifier={self.classifier} | response_mode={self.response_mode}"
        )

    # ------------------------------------------------------------
    # Trial index lookup
    # ------------------------------------------------------------
    def _clean_trial_idx(self, idx):
        idx = np.asarray(idx).ravel()

        if idx.size == 0:
            return np.array([], dtype=int)

        if np.issubdtype(idx.dtype, np.floating):
            idx = idx[~np.isnan(idx)]

        idx = idx.astype(int, copy=False)
        idx = idx[(idx >= 0) & (idx < self.n_trials)]
        return np.unique(idx)

    def _precompute_trial_lookup(self):
        print("Precomputing trial lookup...")
        self.trial_lookup = {}

        for sub in range(self.n_sub):
            self.trial_lookup[sub] = {}
            for category in self.categories:
                pos = self._clean_trial_idx(self.senId[sub][category]["positive"])
                neg = self._clean_trial_idx(self.senId[sub][category]["negative"])

                self.trial_lookup[sub][category] = {
                    "positive": pos,
                    "negative": neg,
                    "neg_minus_pos": {
                        "positive": pos,
                        "negative": neg,
                        "union": (
                            np.unique(np.concatenate([pos, neg]))
                            if (pos.size + neg.size) > 0
                            else np.array([], dtype=int)
                        ),
                    },
                }

    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------
    def _apply_baseline(self, power):
        """
        power: (n_epochs, n_pcs, n_freqs, n_times)
        """
        eps = np.finfo(np.float32).eps
        base = power[..., self.base_mask].mean(axis=-1, keepdims=True)

        if self.baseline_mode == "ratio":
            power = power / np.maximum(base, eps)
        elif self.baseline_mode == "logratio":
            power = np.log10(np.maximum(power, eps) / np.maximum(base, eps))
        elif self.baseline_mode == "percent":
            power = (power - base) / np.maximum(base, eps) * 100.0
        elif self.baseline_mode == "mean":
            power = power - base
        elif self.baseline_mode == "zscore":
            base_std = power[..., self.base_mask].std(axis=-1, keepdims=True)
            power = (power - base) / np.maximum(base_std, eps)
        else:
            raise ValueError(f"Unknown baseline_mode: {self.baseline_mode}")

        return power.astype(self.dtype, copy=False)

    # ------------------------------------------------------------
    # Group handling
    # ------------------------------------------------------------
    def _remap_group_label(self, y_raw, comparison_name):
        spec = self.comparisons[comparison_name]
        if y_raw not in spec["include"]:
            return None
        return spec["map"][int(y_raw)]

    def _get_subjects_for_comparison(self, comparison_name):
        subs, y_bin, y_raw = [], [], []

        for sub in range(self.n_sub):
            yr = self.subIdx[sub]
            if np.isnan(yr):
                continue

            yb = self._remap_group_label(int(yr), comparison_name)
            if yb is None:
                continue

            subs.append(sub)
            y_bin.append(yb)
            y_raw.append(int(yr))

        return (
            np.asarray(subs, dtype=int),
            np.asarray(y_bin, dtype=int),
            np.asarray(y_raw, dtype=int),
        )

    # ------------------------------------------------------------
    # Fold-specific PCA fit on all training trials
    # ------------------------------------------------------------
    def _fit_raw_pca(self, train_subjects):
        blocks = []

        for sub in train_subjects:
            trial_idx = np.arange(self.n_trials, dtype=int)

            if (
                self.pca_trial_cap_per_subject is not None
                and trial_idx.size > self.pca_trial_cap_per_subject
            ):
                trial_idx = self.rng.choice(
                    trial_idx,
                    size=self.pca_trial_cap_per_subject,
                    replace=False,
                )
                trial_idx = np.sort(trial_idx)

            x = self.Dataset[:, :, trial_idx, sub].transpose(2, 1, 0)
            x = x.reshape(-1, self.n_channels)
            blocks.append(x.astype(self.dtype, copy=False))

        if not blocks:
            raise ValueError("No training data available to fit raw PCA.")

        X_raw = np.concatenate(blocks, axis=0)

        mean_ = X_raw.mean(axis=0, keepdims=True)
        X_raw = X_raw - mean_

        if self.raw_pca_standardize:
            std_ = X_raw.std(axis=0, keepdims=True)
            std_[std_ == 0] = 1.0
            X_raw = X_raw / std_
        else:
            std_ = None

        n_components = min(self.numPC, X_raw.shape[0], X_raw.shape[1])
        if n_components < 1:
            raise ValueError("Could not fit at least one PCA component.")

        pca = PCA(n_components=n_components, random_state=self.state)
        pca.fit(X_raw)

        del X_raw, blocks
        gc.collect()

        if std_ is not None:
            std_ = std_.astype(self.dtype, copy=False)

        return mean_.astype(self.dtype, copy=False), std_, pca

    # ------------------------------------------------------------
    # Projection and TF
    # ------------------------------------------------------------
    def _project_subject_to_pcs(self, sub, mean_, std_, pca):
        """
        Returns:
            data_pc: (n_trials, n_pcs, n_times)
        """
        data = np.transpose(self.Dataset[:, :, :, sub], (2, 0, 1)).astype(
            self.dtype, copy=False
        )
        n_trials, n_ch, n_times = data.shape

        flat = data.transpose(0, 2, 1).reshape(-1, n_ch)
        flat = flat - mean_
        if std_ is not None:
            flat = flat / std_

        flat_pc = flat @ pca.components_.T
        data_pc = (
            flat_pc.reshape(n_trials, n_times, -1)
            .transpose(0, 2, 1)
            .astype(self.dtype, copy=False)
        )

        del data, flat, flat_pc
        gc.collect()

        return data_pc

    def _compute_subject_band_power_from_pcs(self, data_pc):
        """
        data_pc: (n_trials, n_pcs, n_times)
        Returns:
            band_power: (n_trials, n_bands, n_pcs, n_decode_times)
        """
        power = mne.time_frequency.tfr_array_morlet(
            data_pc,
            sfreq=self.sfreq,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            output="power",
            zero_mean=True,
            use_fft=True,
            decim=1,
            n_jobs=self.tf_n_jobs,
        ).astype(self.dtype, copy=False)

        power = self._apply_baseline(power)
        power = power[..., self.decode_mask]

        n_trials, n_pcs, _, n_decode_times = power.shape
        n_bands = len(self.band_names)

        band_power = np.empty(
            (n_trials, n_bands, n_pcs, n_decode_times),
            dtype=self.dtype,
        )

        for bi, band_name in enumerate(self.band_names):
            f_lo, f_hi = self.bands[band_name]
            f_mask = (self.freqs >= f_lo) & (self.freqs < f_hi)

            if not np.any(f_mask):
                raise ValueError(
                    f"No frequencies for band {band_name}: {f_lo}-{f_hi} Hz"
                )

            band_power[:, bi, :, :] = power[:, :, f_mask, :].mean(axis=2)

        del power
        gc.collect()

        return band_power

    def _compute_evoked_map_from_pcs(self, data_pc_avg):
        """
        data_pc_avg: (n_pcs, n_times)
        Returns:
            subject_map: (n_bands, n_pcs, n_decode_times)
        """
        power = mne.time_frequency.tfr_array_morlet(
            data_pc_avg[np.newaxis, ...],
            sfreq=self.sfreq,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            output="power",
            zero_mean=True,
            use_fft=True,
            decim=1,
            n_jobs=self.tf_n_jobs,
        ).astype(self.dtype, copy=False)

        power = self._apply_baseline(power)
        power = power[..., self.decode_mask]  # 1 x n_pcs x n_freqs x n_decode_times
        power = power[0]

        n_pcs, _, n_decode_times = power.shape
        n_bands = len(self.band_names)

        subject_map = np.empty(
            (n_bands, n_pcs, n_decode_times),
            dtype=self.dtype,
        )

        for bi, band_name in enumerate(self.band_names):
            f_lo, f_hi = self.bands[band_name]
            f_mask = (self.freqs >= f_lo) & (self.freqs < f_hi)

            if not np.any(f_mask):
                raise ValueError(
                    f"No frequencies for band {band_name}: {f_lo}-{f_hi} Hz"
                )

            subject_map[bi, :, :] = power[:, f_mask, :].mean(axis=1)

        del power
        gc.collect()

        return subject_map

    def _build_fold_cache(self, subject_ids, mean_, std_, pca):
        """
        Compute fold-specific representation once per subject.
        For both modes, cache projected PC trials:
            cache[sub] = data_pc with shape (n_trials, n_pcs, n_times)
        """
        cache = {}
        for sub in tqdm(subject_ids, desc="Building fold cache", leave=False):
            cache[sub] = self._project_subject_to_pcs(sub, mean_, std_, pca)
        return cache

    # ------------------------------------------------------------
    # Build subject maps from fold cache
    # ------------------------------------------------------------
    def _build_subject_maps(self, subject_ids, category, contrast, fold_cache):
        """
        Returns:
            X: (n_subjects_valid, n_bands, n_pcs, n_times)
            keep_mask: bool mask aligned to subject_ids
        """
        X_list = []
        keep_mask = np.zeros(subject_ids.size, dtype=bool)

        for i, sub in enumerate(subject_ids):
            data_pc = fold_cache[sub]  # (n_trials, n_pcs, n_times)

            if contrast == "positive":
                pos_idx = self.trial_lookup[sub][category]["positive"]
                if pos_idx.size == 0:
                    continue

                if self.response_mode == "induced":
                    band_power = self._compute_subject_band_power_from_pcs(
                        data_pc[pos_idx]
                    )
                    subject_map = band_power.mean(axis=0)
                else:  # evoked
                    data_pc_avg = data_pc[pos_idx].mean(axis=0)
                    subject_map = self._compute_evoked_map_from_pcs(data_pc_avg)

            elif contrast == "negative":
                neg_idx = self.trial_lookup[sub][category]["negative"]
                if neg_idx.size == 0:
                    continue

                if self.response_mode == "induced":
                    band_power = self._compute_subject_band_power_from_pcs(
                        data_pc[neg_idx]
                    )
                    subject_map = band_power.mean(axis=0)
                else:  # evoked
                    data_pc_avg = data_pc[neg_idx].mean(axis=0)
                    subject_map = self._compute_evoked_map_from_pcs(data_pc_avg)

            elif contrast == "neg_minus_pos":
                pos_idx = self.trial_lookup[sub][category]["neg_minus_pos"]["positive"]
                neg_idx = self.trial_lookup[sub][category]["neg_minus_pos"]["negative"]
                if pos_idx.size == 0 or neg_idx.size == 0:
                    continue

                if self.response_mode == "induced":
                    band_power_pos = self._compute_subject_band_power_from_pcs(
                        data_pc[pos_idx]
                    )
                    band_power_neg = self._compute_subject_band_power_from_pcs(
                        data_pc[neg_idx]
                    )
                    subject_map = band_power_neg.mean(axis=0) - band_power_pos.mean(
                        axis=0
                    )
                else:  # evoked
                    data_pc_avg_pos = data_pc[pos_idx].mean(axis=0)
                    data_pc_avg_neg = data_pc[neg_idx].mean(axis=0)
                    subject_map = self._compute_evoked_map_from_pcs(
                        data_pc_avg_neg
                    ) - self._compute_evoked_map_from_pcs(data_pc_avg_pos)

            else:
                raise ValueError(f"Unknown contrast: {contrast}")

            X_list.append(subject_map.astype(self.dtype, copy=False))
            keep_mask[i] = True

        if not X_list:
            return None, keep_mask

        X = np.stack(X_list, axis=0)
        return X, keep_mask

    # ------------------------------------------------------------
    # Feature scaling for classifier
    # ------------------------------------------------------------
    @staticmethod
    def _scale_train_test(X_train, X_test):
        mean_ = X_train.mean(axis=0, keepdims=True)
        std_ = X_train.std(axis=0, keepdims=True)
        std_[std_ == 0] = 1.0
        return (X_train - mean_) / std_, (X_test - mean_) / std_

    # ------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------
    def _make_classifier(self):
        if self.classifier == "svm":
            return LinearSVC(
                C=self.clf_c,
                class_weight="balanced",
                random_state=self.state,
                max_iter=10000,
            )

        if self.classifier == "logreg":
            return LogisticRegression(
                C=self.clf_c,
                penalty="l2",
                class_weight="balanced",
                solver="liblinear",
                random_state=self.state,
                max_iter=10000,
            )

        raise ValueError(f"Unknown classifier: {self.classifier}")

    # ------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------
    def run(self):
        self.load_EEG()
        self._precompute_trial_lookup()

        results = {
            "meta": {
                "sfreq": self.sfreq,
                "times": self.decode_times,
                "bands": self.bands,
                "band_names": self.band_names,
                "categories": self.categories,
                "contrasts": self.contrasts,
                "comparisons": self.comparisons,
                "numPC": self.numPC,
                "decode_tmin": self.decode_tmin,
                "decode_tmax": self.decode_tmax,
                "classifier": self.classifier,
                "response_mode": self.response_mode,
            }
        }

        for comparison_name in self.comparisons:
            print("\n" + "=" * 90)
            print(f"Comparison: {comparison_name}")
            print("=" * 90)

            comparison_subjects, comparison_y, comparison_y_raw = (
                self._get_subjects_for_comparison(comparison_name)
            )

            classes, counts = np.unique(comparison_y, return_counts=True)
            if len(classes) < 2:
                print(f"Skipping {comparison_name}: less than 2 classes.")
                continue

            min_class_count = counts.min()
            n_splits = min(self.kfold, min_class_count)
            if n_splits < 2:
                print(
                    f"Skipping {comparison_name}: smallest class has only {min_class_count} subjects."
                )
                continue

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.state,
            )

            results[comparison_name] = {}

            for category in self.categories:
                results[comparison_name][category] = {}
                for contrast in self.contrasts:
                    results[comparison_name][category][contrast] = {
                        "fold_scores": np.full(
                            (len(self.band_names), self.decode_times.size, n_splits),
                            np.nan,
                            dtype=self.dtype,
                        ),
                        "n_splits_used": n_splits,
                    }

            for fi, (train_idx, test_idx) in enumerate(
                cv.split(comparison_subjects, comparison_y)
            ):
                print(f"\nFold {fi + 1}/{n_splits}")

                train_subjects = comparison_subjects[train_idx]
                test_subjects = comparison_subjects[test_idx]
                y_train_full = comparison_y[train_idx]
                y_test_full = comparison_y[test_idx]

                mean_, std_, pca = self._fit_raw_pca(train_subjects)

                fold_subjects = np.concatenate([train_subjects, test_subjects])
                fold_cache = self._build_fold_cache(fold_subjects, mean_, std_, pca)

                for category in self.categories:
                    for contrast in self.contrasts:
                        X_train, keep_train = self._build_subject_maps(
                            train_subjects, category, contrast, fold_cache
                        )
                        X_test, keep_test = self._build_subject_maps(
                            test_subjects, category, contrast, fold_cache
                        )

                        if X_train is None or X_test is None:
                            continue

                        y_train = y_train_full[keep_train]
                        y_test = y_test_full[keep_test]

                        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                            continue

                        for bi, band_name in enumerate(self.band_names):
                            for ti in range(self.decode_times.size):
                                Xt_train = X_train[:, bi, :, ti]
                                Xt_test = X_test[:, bi, :, ti]

                                Xt_train, Xt_test = self._scale_train_test(
                                    Xt_train, Xt_test
                                )

                                clf = self._make_classifier()
                                clf.fit(Xt_train, y_train)
                                y_pred = clf.predict(Xt_test)

                                results[comparison_name][category][contrast][
                                    "fold_scores"
                                ][bi, ti, fi] = balanced_accuracy_score(y_test, y_pred)

                del mean_, std_, pca, fold_cache
                gc.collect()

            for category in self.categories:
                for contrast in self.contrasts:
                    fold_scores = results[comparison_name][category][contrast][
                        "fold_scores"
                    ]
                    results[comparison_name][category][contrast]["score"] = np.nanmean(
                        fold_scores, axis=2
                    )
                    results[comparison_name][category][contrast]["n_subjects"] = int(
                        comparison_subjects.size
                    )
                    results[comparison_name][category][contrast][
                        "labels_binary"
                    ] = comparison_y
                    results[comparison_name][category][contrast][
                        "labels_original"
                    ] = comparison_y_raw
                    results[comparison_name][category][contrast][
                        "used_subjects"
                    ] = comparison_subjects

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, self.saveName)

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

        print(f"\nSaved results to:\n{save_path}")


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

    comparisons = {
        "control_vs_depressed": {"include": [1, 2], "map": {1: 0, 2: 1}},
        "depressed_vs_suicidal": {"include": [2, 3], "map": {2: 0, 3: 1}},
        "control_vs_suicidal": {"include": [1, 3], "map": {1: 0, 3: 1}},
        "control_vs_depressedsuicidal": {
            "include": [1, 2, 3],
            "map": {1: 0, 2: 1, 3: 1},
        },
    }

    # # Induced
    # decoder_induced = TFBandDecoder(
    #     tmin=-0.3,
    #     tmax=1.5,
    #     decode_tmin=-0.2,
    #     decode_tmax=1.0,
    #     numPC=3,
    #     bands=bands,
    #     comparisons=comparisons,
    #     classifier="logreg",
    #     response_mode="induced",
    #     saveName="TFclassDecoding_logreg_induced.pkl",
    #     tf_n_jobs=1,
    # )

    # Evoked
    decoder_evoked = TFBandDecoder(
        tmin=-0.3,
        tmax=1.5,
        decode_tmin=-0.2,
        decode_tmax=1.0,
        numPC=3,
        bands=bands,
        comparisons=comparisons,
        classifier="svm",
        response_mode="evoked",
        saveName="TFclassDecoding_svm_evoked.pkl",
        tf_n_jobs=1,
    )
