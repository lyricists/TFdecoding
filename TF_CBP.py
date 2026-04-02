# ============================================================
# Cluster-based permutation test for time-frequency data
# ============================================================


import numpy as np
from mne.stats import permutation_cluster_test, combine_adjacency
from scipy.stats import ttest_ind
from scipy.stats import t


def stat_fun(x, y):
    tvals, _ = ttest_ind(x, y, axis=0, equal_var=False)
    return tvals


def cluster_based_permutation_test(data1, data2, n_permutations=2000, threshold_p=0.05):
    """
    Perform a cluster-based permutation test on time-frequency data.

    Parameters
    ----------
    data1 : array-like, shape (n_subjects, n_times, n_freqs)
        Time-frequency data for the first condition.
    data2 : array-like, shape (n_subjects, n_times, n_freqs)
        Time-frequency data for the second condition.
    n_permutations : int, optional
        Number of permutations to perform (default is 2000).
    threshold_p : float, optional
        P-value threshold for forming clusters (default is 0.05).

    Returns
    -------
    sig_mask : array, shape (n_freqs, n_times)
        Boolean mask indicating significant clusters (True for significant).
    T_obs : array, shape (n_times, n_freqs)
        Observed t-values for each time-frequency point.
    clusters : list of arrays
        List of clusters found in the observed data.
    cluster_p_values : array
        P-values for each cluster."""

    n_time = data1.shape[1]
    n_freq = data2.shape[2]

    adjacency = combine_adjacency(n_time, n_freq)

    if threshold_p is not None:
        df = data1.shape[0] + data2.shape[0] - 2
        threshold = t.ppf(1 - threshold_p / 2, df)  # two-sided cluster-forming p=.05
    else:
        threshold = None

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [data1, data2],
        stat_fun=stat_fun,
        adjacency=adjacency,
        n_permutations=n_permutations,
        tail=0,
        threshold=threshold,
        out_type="mask",
        seed=42,
        n_jobs=-1,
    )

    # clusters and T_obs come out as (time, freq)
    sig_mask = np.zeros((n_time, n_freq), dtype=bool)

    for clu, p in zip(clusters, cluster_p_values):
        if p < 0.05:
            sig_mask |= clu

    # transpose to (freq, time) for imshow
    sig_mask = sig_mask.T
    T_obs = T_obs.T

    return sig_mask, T_obs, cluster_p_values, clusters
