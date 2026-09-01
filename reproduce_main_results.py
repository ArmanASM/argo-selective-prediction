#!/usr/bin/env python3
"""Reproduce the primary nested LOPO and selective-prediction results.

This script uses the repository's included ARGO-derived 750 ms bipolar feature
matrix and annotation audit. It does not require redistribution of the raw
PhysioNet waveforms.
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

PARAMS = [
    {"name": "L7_m20", "num_leaves": 7, "min_child_samples": 20},
    {"name": "L15_m20", "num_leaves": 15, "min_child_samples": 20},
    {"name": "L31_m20", "num_leaves": 31, "min_child_samples": 20},
    {"name": "L15_m40", "num_leaves": 15, "min_child_samples": 40},
]


def make_model(cfg):
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def ece(y, p, n_bins=10):
    total = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return total


def load_data(repo_root):
    audit = pd.read_csv(repo_root / "data/audit/annotation_audit.csv")
    feat = pd.read_csv(repo_root / "data/derived/bipolar_last750ms_features.csv")
    df = audit.merge(feat, on=["patient", "record"], how="inner")
    df = df[df.consensus_label.isin(["A", "P"])].reset_index(drop=True)
    cols = [c for c in feat.columns if c not in ("patient", "record")]
    X = df[cols].to_numpy(float)
    y = (df.consensus_label == "A").astype(int).to_numpy()
    pts = df.patient.to_numpy()
    rec = df.record.to_numpy()
    patients = sorted(df.patient.unique(), key=lambda s: int(s[2:]))
    return audit, feat, df, cols, X, y, pts, rec, patients


def nested_lopo(repo_root, out):
    audit, feat, df, cols, X, y, pts, rec, patients = load_data(repo_root)
    fold_rows, pred_rows, tune_rows = [], [], []

    for outer_p in patients:
        test = pts == outer_p
        train = ~test
        tr_idx = np.where(train)[0]
        inner_patients = [p for p in patients if p != outer_p]
        scores, oofs = [], {}

        for cfg in PARAMS:
            oof = np.full(train.sum(), np.nan)
            bas, aucs = [], []
            for inner_p in inner_patients:
                iva = pts[tr_idx] == inner_p
                itr = ~iva
                model = make_model(cfg)
                model.fit(X[tr_idx][itr], y[tr_idx][itr])
                prob = model.predict_proba(X[tr_idx][iva])[:, 1]
                oof[iva] = prob
                yy = y[tr_idx][iva]
                bas.append(balanced_accuracy_score(yy, prob >= 0.5))
                aucs.append(roc_auc_score(yy, prob))
            mean_ba, mean_auc = float(np.mean(bas)), float(np.mean(aucs))
            scores.append((mean_ba, mean_auc, cfg["name"]))
            oofs[cfg["name"]] = oof
            tune_rows.append({
                "outer_patient": outer_p,
                "config": cfg["name"],
                "inner_macro_balanced_accuracy": mean_ba,
                "inner_macro_auroc": mean_auc,
            })

        scores.sort(reverse=True)
        best_name = scores[0][2]
        cfg = next(c for c in PARAMS if c["name"] == best_name)
        oof = oofs[best_name]

        # Platt scaling fit only on inner out-of-fold predictions.
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(logit(np.clip(oof, 1e-5, 1 - 1e-5)).reshape(-1, 1), y[tr_idx])

        model = make_model(cfg)
        model.fit(X[train], y[train])
        raw = model.predict_proba(X[test])[:, 1]
        platt = cal.predict_proba(logit(np.clip(raw, 1e-5, 1 - 1e-5)).reshape(-1, 1))[:, 1]
        yy = y[test]

        for version, prob in [("raw", raw), ("platt", platt)]:
            pred = prob >= 0.5
            fold_rows.append({
                "patient": outer_p,
                "version": version,
                "selected_config": best_name,
                "n": len(yy),
                "accuracy": accuracy_score(yy, pred),
                "balanced_accuracy": balanced_accuracy_score(yy, pred),
                "f1": f1_score(yy, pred),
                "auroc": roc_auc_score(yy, prob),
                "brier": brier_score_loss(yy, prob),
                "ece10": ece(yy, prob),
            })

        for r, target, p_raw, p_platt in zip(rec[test], yy, raw, platt):
            pred_rows.append({
                "patient": outer_p,
                "record": r,
                "y": int(target),
                "pA_raw": float(p_raw),
                "pA_platt": float(p_platt),
                "pred_raw": int(p_raw >= 0.5),
                "pred_platt": int(p_platt >= 0.5),
                "selected_config": best_name,
            })
        print(f"Completed {outer_p}: {best_name}")

    fold = pd.DataFrame(fold_rows)
    pred = pd.DataFrame(pred_rows)
    tune = pd.DataFrame(tune_rows)
    fold.to_csv(out / "nested_lopo_fold_metrics.csv", index=False)
    pred.to_csv(out / "nested_lopo_predictions.csv", index=False)
    tune.to_csv(out / "nested_tuning_scores.csv", index=False)
    return audit, feat, df, cols, X, y, pts, rec, patients, fold, pred, tune


def aggregate(out, audit, fold, pred, patients):
    rows = []
    for version in ["raw", "platt"]:
        prob = pred[f"pA_{version}"].to_numpy()
        y = pred.y.to_numpy()
        hard = prob >= 0.5
        f = fold[fold.version == version]
        rows.append({
            "version": version,
            "pooled_accuracy": accuracy_score(y, hard),
            "pooled_balanced_accuracy": balanced_accuracy_score(y, hard),
            "pooled_f1": f1_score(y, hard),
            "pooled_auroc": roc_auc_score(y, prob),
            "brier": brier_score_loss(y, prob),
            "ece10": ece(y, prob),
            "macro_patient_accuracy": f.accuracy.mean(),
            "macro_patient_balanced_accuracy": f.balanced_accuracy.mean(),
            "macro_patient_auroc": f.auroc.mean(),
            "macro_patient_brier": f.brier.mean(),
            "macro_patient_ece10": f.ece10.mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "nested_calibration_summary.csv", index=False)

    # Patient-cluster bootstrap, preserving the original study seed.
    rng = np.random.default_rng(20260827)
    groups = {
        p: (pred.loc[pred.patient == p, "y"].to_numpy(dtype=int),
            pred.loc[pred.patient == p, "pA_raw"].to_numpy(dtype=float))
        for p in patients
    }
    fraw = fold[fold.version == "raw"].set_index("patient")
    macro_ba = {p: float(fraw.loc[p, "balanced_accuracy"]) for p in patients}
    boot = np.empty((10000, 4), dtype=float)
    for b in range(10000):
        sample = rng.choice(patients, len(patients), replace=True)
        yy = np.concatenate([groups[p][0] for p in sample])
        prob = np.concatenate([groups[p][1] for p in sample])
        hard = prob >= 0.5
        boot[b, 0] = accuracy_score(yy, hard)
        boot[b, 1] = balanced_accuracy_score(yy, hard)
        boot[b, 2] = roc_auc_score(yy, prob)
        boot[b, 3] = np.mean([macro_ba[p] for p in sample])
    raw = summary[summary.version == "raw"].iloc[0]
    metric_map = [
        ("pooled_accuracy", raw.pooled_accuracy),
        ("pooled_balanced_accuracy", raw.pooled_balanced_accuracy),
        ("pooled_auroc", raw.pooled_auroc),
        ("macro_patient_balanced_accuracy", raw.macro_patient_balanced_accuracy),
    ]
    ci_rows = []
    for j, (metric, point) in enumerate(metric_map):
        ci_rows.append({
            "metric": metric,
            "point": point,
            "cluster_boot_95_low": np.quantile(boot[:, j], 0.025),
            "cluster_boot_95_high": np.quantile(boot[:, j], 0.975),
        })
    pd.DataFrame(ci_rows).to_csv(out / "nested_cluster_bootstrap_ci.csv", index=False)

    cal_rows = []
    for metric in ["brier", "ece10", "accuracy", "balanced_accuracy"]:
        r = fold[fold.version == "raw"].set_index("patient")[metric]
        c = fold[fold.version == "platt"].set_index("patient")[metric]
        delta = (c - r).values
        try:
            test = stats.wilcoxon(delta, method="exact")
        except Exception:
            test = stats.wilcoxon(delta)
        cal_rows.append({
            "metric": metric,
            "mean_platt_minus_raw": delta.mean(),
            "median_delta": np.median(delta),
            "wilcoxon_two_sided_p": test.pvalue,
        })
    pd.DataFrame(cal_rows).to_csv(out / "calibration_patient_stats.csv", index=False)
    return summary


def prospective_selective(out, audit, df, cols, X, y, pts, patients, fold, pred):
    raw_folds = fold[fold.version == "raw"]
    cfg_map = {"L7_m20": (7, 20), "L15_m20": (15, 20), "L31_m20": (31, 20), "L15_m40": (15, 40)}
    coverages = [0.9, 0.8, 0.7, 0.6, 0.5]
    rows = []

    for outer_p in patients:
        name = raw_folds.loc[raw_folds.patient == outer_p, "selected_config"].iloc[0]
        leaves, min_child = cfg_map[name]
        train = pts != outer_p
        tr_idx = np.where(train)[0]
        train_patients = [p for p in patients if p != outer_p]
        oof = np.full(train.sum(), np.nan)

        for inner_p in train_patients:
            iva = pts[tr_idx] == inner_p
            itr = ~iva
            cfg = {"num_leaves": leaves, "min_child_samples": min_child}
            model = LGBMClassifier(
                n_estimators=180, learning_rate=0.04,
                num_leaves=leaves, min_child_samples=min_child,
                colsample_bytree=0.8, reg_lambda=1.0,
                class_weight="balanced", random_state=42,
                verbosity=-1, n_jobs=-1,
            )
            model.fit(X[tr_idx][itr], y[tr_idx][itr])
            oof[iva] = model.predict_proba(X[tr_idx][iva])[:, 1]

        train_conf = np.maximum(oof, 1 - oof)
        g = pred[pred.patient == outer_p].copy().merge(
            audit[["patient", "record", "n_unique"]], on=["patient", "record"], how="left"
        )
        g["confidence"] = np.maximum(g.pA_raw, 1 - g.pA_raw)
        full_acc = accuracy_score(g.y, g.pred_raw)

        for cov in coverages:
            threshold = float(np.quantile(train_conf, 1 - cov, method="linear"))
            keep = g[g.confidence >= threshold]
            abstain = g[g.confidence < threshold]
            ba = balanced_accuracy_score(keep.y, keep.pred_raw) if keep.y.nunique() > 1 else np.nan
            auc = roc_auc_score(keep.y, keep.pA_raw) if keep.y.nunique() > 1 else np.nan
            acc = accuracy_score(keep.y, keep.pred_raw) if len(keep) else np.nan
            rows.append({
                "patient": outer_p,
                "target_coverage": cov,
                "threshold_from_training_oof": threshold,
                "actual_test_coverage": len(keep) / len(g),
                "retained_n": len(keep),
                "abstained_n": len(abstain),
                "full_accuracy": full_acc,
                "retained_accuracy": acc,
                "accuracy_gain": acc - full_acc,
                "retained_balanced_accuracy": ba,
                "retained_auroc": auc,
                "retained_disagreement": (keep.n_unique > 1).mean() if len(keep) else np.nan,
                "abstained_disagreement": (abstain.n_unique > 1).mean() if len(abstain) else np.nan,
                "disagreement_enrichment": ((abstain.n_unique > 1).mean() - (keep.n_unique > 1).mean()) if len(abstain) and len(keep) else np.nan,
            })

    by_patient = pd.DataFrame(rows)
    by_patient.to_csv(out / "prospective_selective_by_patient.csv", index=False)

    summaries, stat_rows = [], []
    rng = np.random.default_rng(42)
    for cov in coverages:
        q = by_patient[by_patient.target_coverage == cov]
        kept = []
        for _, rr in q.iterrows():
            g = pred[pred.patient == rr.patient].copy()
            g["confidence"] = np.maximum(g.pA_raw, 1 - g.pA_raw)
            kept.append(g[g.confidence >= rr.threshold_from_training_oof])
        k = pd.concat(kept, ignore_index=True)
        yy = k.y.to_numpy(); prob = k.pA_raw.to_numpy(); hard = k.pred_raw.to_numpy()
        summaries.append({
            "target_coverage": cov,
            "actual_pooled_coverage": len(k) / len(pred),
            "retained_n": len(k),
            "pooled_accuracy": accuracy_score(yy, hard),
            "pooled_balanced_accuracy": balanced_accuracy_score(yy, hard),
            "pooled_auroc": roc_auc_score(yy, prob),
            "mean_patient_accuracy_gain": q.accuracy_gain.mean(),
            "mean_disagreement_enrichment": q.disagreement_enrichment.mean(),
        })
        for metric in ["accuracy_gain", "disagreement_enrichment"]:
            vals = q[metric].dropna().values
            try:
                w = stats.wilcoxon(vals, alternative="greater", method="exact")
            except Exception:
                w = stats.wilcoxon(vals, alternative="greater")
            bb = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(10000)])
            stat_rows.append({
                "target_coverage": cov,
                "quantity": metric,
                "mean": vals.mean(),
                "median": np.median(vals),
                "one_sided_wilcoxon_p": w.pvalue,
                "boot95_low": np.quantile(bb, 0.025),
                "boot95_high": np.quantile(bb, 0.975),
            })
    pd.DataFrame(summaries).to_csv(out / "prospective_selective_summary.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out / "prospective_selective_stats.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = (args.output or (repo / "reproduced_results")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    audit, feat, df, cols, X, y, pts, rec, patients, fold, pred, tune = nested_lopo(repo, out)
    summary = aggregate(out, audit, fold, pred, patients)
    prospective_selective(out, audit, df, cols, X, y, pts, patients, fold, pred)

    print("\nPrimary nested-LOPO summary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
