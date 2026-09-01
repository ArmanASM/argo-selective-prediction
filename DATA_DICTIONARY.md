# Data Dictionary

## `data/derived/bipolar_last750ms_features.csv`

Each row corresponds to one ARGO mapping-point recording. `patient` and `record` are source identifiers; the remaining 43 columns are predictors computed from the final 750 ms of the bipolar EGM.

| Column | Description |
|---|---|
| `mean` | Mean of mean-centered bipolar EGM (numerical near-zero check). |
| `std` | Standard deviation of bipolar EGM amplitude. |
| `rms` | Root-mean-square amplitude. |
| `ptp` | Peak-to-peak amplitude. |
| `max_abs` | Maximum absolute amplitude. |
| `median_abs` | Median absolute amplitude. |
| `abs_q75` | 75th percentile of absolute amplitude. |
| `abs_q90` | 90th percentile of absolute amplitude. |
| `abs_q95` | 95th percentile of absolute amplitude. |
| `abs_q99` | 99th percentile of absolute amplitude. |
| `mad` | Median absolute deviation. |
| `skew` | Sample skewness. |
| `kurtosis` | Excess kurtosis. |
| `line_length` | Sum of absolute first differences. |
| `mean_abs_diff` | Mean absolute first difference. |
| `std_diff` | Standard deviation of first differences. |
| `max_abs_diff` | Maximum absolute first difference. |
| `zero_cross_rate` | Fraction of adjacent sample pairs with a strict sign change (`x[t] * x[t+1] < 0`), so zero-valued transitions are not counted. |
| `hjorth_mobility` | Hjorth mobility. |
| `hjorth_complexity` | Hjorth complexity. |
| `peaks_gt_0.1max` | Count of absolute-amplitude peaks above 10% of maximum. |
| `peaks_gt_0.2max` | Count of absolute-amplitude peaks above 20% of maximum. |
| `peaks_gt_0.3max` | Count of absolute-amplitude peaks above 30% of maximum. |
| `peaks_gt_0.5max` | Count of absolute-amplitude peaks above 50% of maximum. |
| `autocorr_1` | Autocorrelation at lag 1 sample. |
| `autocorr_5` | Autocorrelation at lag 5 samples. |
| `autocorr_10` | Autocorrelation at lag 10 samples. |
| `autocorr_20` | Autocorrelation at lag 20 samples. |
| `autocorr_50` | Autocorrelation at lag 50 samples. |
| `autocorr_100` | Autocorrelation at lag 100 samples. |
| `spectral_entropy` | Normalized entropy of Welch power spectrum. |
| `spectral_centroid` | Power-weighted spectral centroid. |
| `median_freq` | Frequency below which 50% of spectral power lies. |
| `dominant_freq` | Frequency bin with maximum spectral power. |
| `bandpower_16_40` | Integrated Welch power from 16–40 Hz. |
| `bandpower_40_80` | Integrated Welch power from 40–80 Hz. |
| `bandpower_80_125` | Integrated Welch power from 80–125 Hz. |
| `bandpower_125_250` | Integrated Welch power from 125–250 Hz. |
| `bandpower_250_500` | Integrated Welch power from 250–500 Hz. |
| `hf_lf_ratio` | Power ratio: 80–500 Hz divided by 16–80 Hz. |
| `veryhf_ratio` | 250–500 Hz power divided by total 16–500 Hz band power. |
| `teager_mean_abs` | Mean absolute Teager energy operator value. |
| `teager_std` | Standard deviation of Teager energy operator value. |

## `data/audit/annotation_audit.csv`

This table contains the parsed three independent expert annotations, consensus annotation, AVP delineation fields, and derived disagreement variables used in the study. Key fields include `ann1_label`, `ann2_label`, `ann3_label`, `consensus_label`, `n_unique`, `all_agree`, `two_one_split`, `three_way_split`, and `expert_entropy`.

## `results/`

The `results/` directory contains frozen analysis outputs reported in the manuscript, including nested-LOPO fold metrics/predictions, calibration results, patient-cluster bootstrap confidence intervals, prospective selective-prediction results, temporal-window sensitivity, and permutation feature importance.