#!/usr/bin/env python3
"""Extract the 43 bipolar-EGM features used in the primary 750 ms model.

Requires the ARGO v1.0.0 ZIP downloaded from PhysioNet. Raw ARGO data are not
redistributed with this repository.
"""
from pathlib import Path
import argparse, zipfile, re, math
import numpy as np
import pandas as pd
from scipy import stats, signal


def safe_skew(x):
    v = stats.skew(x, bias=False)
    return 0.0 if not np.isfinite(v) else float(v)


def safe_kurt(x):
    v = stats.kurtosis(x, fisher=True, bias=False)
    return 0.0 if not np.isfinite(v) else float(v)


def autocorr(x, lag):
    if len(x) <= lag:
        return 0.0
    denom = float(np.sum(x * x))
    return 0.0 if denom < 1e-12 else float(np.sum(x[:-lag] * x[lag:]) / denom)


def bandpower(freq, power, lo, hi):
    mask = (freq >= lo) & (freq < hi)
    return 0.0 if not mask.any() else float(np.trapezoid(power[mask], freq[mask]))


def features(x, fs=1000.0):
    x = np.asarray(x, float)
    x = x - x.mean()
    dx = np.diff(x)
    absx = np.abs(x)
    eps = 1e-12
    maxa = float(absx.max()) if len(absx) else 0.0
    peak_counts = {}
    for frac in (0.1, 0.2, 0.3, 0.5):
        peak_counts[frac] = 0 if maxa <= eps else len(signal.find_peaks(absx, height=frac * maxa, distance=3)[0])

    f, p = signal.welch(x, fs=fs, nperseg=min(256, len(x)), noverlap=min(128, max(0, len(x)//2 - 1)), scaling="density")
    ps = p.sum(); pn = p / ps if ps > eps else np.zeros_like(p)
    spec_entropy = float(-(pn[pn > 0] * np.log2(pn[pn > 0])).sum() / np.log2(len(pn))) if (pn > 0).sum() > 1 else 0.0
    centroid = float((f * p).sum() / (p.sum() + eps))
    cdf = np.cumsum(p)
    median_freq = float(f[np.searchsorted(cdf, cdf[-1] / 2)]) if cdf[-1] > 0 else 0.0
    dominant_freq = float(f[np.argmax(p)]) if len(p) else 0.0
    total_power = float(np.trapezoid(p, f))
    norm = total_power if total_power > eps else 1.0
    b1 = bandpower(f, p, 16, 40) / norm; b2 = bandpower(f, p, 40, 80) / norm; b3 = bandpower(f, p, 80, 125) / norm
    b4 = bandpower(f, p, 125, 250) / norm; b5 = bandpower(f, p, 250, 500) / norm
    lf, hf = b1 + b2, b3 + b4
    teager = x[1:-1]**2 - x[:-2]*x[2:] if len(x) >= 3 else np.array([0.0])
    varx = np.var(x); vard = np.var(dx) if len(dx) else 0.0
    mobility = math.sqrt(vard / (varx + eps))
    dd = np.diff(dx)
    mobility_d = math.sqrt(np.var(dd) / (vard + eps)) if len(dd) else 0.0
    complexity = mobility_d / (mobility + eps)

    return {
        "mean": float(x.mean()), "std": float(x.std()), "rms": float(np.sqrt(np.mean(x*x))),
        "ptp": float(np.ptp(x)), "max_abs": maxa, "median_abs": float(np.median(absx)),
        "abs_q75": float(np.quantile(absx, .75)), "abs_q90": float(np.quantile(absx, .90)),
        "abs_q95": float(np.quantile(absx, .95)), "abs_q99": float(np.quantile(absx, .99)),
        "mad": float(np.median(np.abs(x - np.median(x)))), "skew": safe_skew(x), "kurtosis": safe_kurt(x),
        "line_length": float(np.abs(dx).sum()), "mean_abs_diff": float(np.mean(np.abs(dx))),
        "std_diff": float(np.std(dx)), "max_abs_diff": float(np.max(np.abs(dx))),
        "zero_cross_rate": float(np.mean((x[:-1] * x[1:]) < 0)),
        "hjorth_mobility": mobility, "hjorth_complexity": complexity,
        "peaks_gt_0.1max": peak_counts[.1], "peaks_gt_0.2max": peak_counts[.2],
        "peaks_gt_0.3max": peak_counts[.3], "peaks_gt_0.5max": peak_counts[.5],
        "autocorr_1": autocorr(x, 1), "autocorr_5": autocorr(x, 5), "autocorr_10": autocorr(x, 10),
        "autocorr_20": autocorr(x, 20), "autocorr_50": autocorr(x, 50), "autocorr_100": autocorr(x, 100),
        "spectral_entropy": spec_entropy, "spectral_centroid": centroid,
        "median_freq": median_freq, "dominant_freq": dominant_freq,
        "bandpower_16_40": b1, "bandpower_40_80": b2, "bandpower_80_125": b3,
        "bandpower_125_250": b4, "bandpower_250_500": b5,
        "hf_lf_ratio": float(hf / (lf + eps)), "veryhf_ratio": float(b5 / (lf + hf + eps)),
        "teager_mean_abs": float(np.mean(np.abs(teager))), "teager_std": float(np.std(teager)),
    }


def parse_gain(header_text):
    # First signal line, matching the original analysis.
    line = header_text.strip().splitlines()[1].split()
    match = re.match(r"([0-9.+-eE]+)", line[2])
    return float(match.group(1)) if match else 1.0


def detect_root(names):
    candidates = [n.split('ARGODataset_Folder/')[0] for n in names if 'ARGODataset_Folder/' in n]
    if not candidates:
        raise RuntimeError("Could not locate ARGODataset_Folder in ZIP")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--argo-zip', required=True, type=Path)
    ap.add_argument('--output', type=Path, default=Path('data/derived/bipolar_last750ms_features.csv'))
    args = ap.parse_args()

    rows = []
    with zipfile.ZipFile(args.argo_zip) as z:
        names = z.namelist()
        root = detect_root(names)
        headers = [n for n in names if re.search(r'ARGODataset_Folder/Pt\d+/P\d+\.hea$', n)]
        headers.sort(key=lambda n: (int(re.search(r'/Pt(\d+)/', n).group(1)), int(re.search(r'/P(\d+)\.hea$', n).group(1))))
        for i, hname in enumerate(headers):
            patient = re.search(r'/Pt(\d+)/', hname).group(1)
            record = re.search(r'/(P\d+)\.hea$', hname).group(1)
            base = hname[:-4]
            header = z.read(hname).decode(errors='replace')
            gain = parse_gain(header)
            raw = np.frombuffer(z.read(base + '.dat'), dtype='<i4').reshape(-1, 15)
            bipolar = raw[:, 0] / gain
            row = {'patient': f'Pt{patient}', 'record': record}
            row.update(features(bipolar[-750:]))
            rows.append(row)
            if (i + 1) % 250 == 0:
                print(f'Extracted {i+1}/{len(headers)}')

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Wrote {len(rows)} records to {out}')


if __name__ == '__main__':
    main()
