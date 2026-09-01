# Reproducibility Guide

## 1. Environment

Python 3.11+ is recommended. Install dependencies with `pip install -r requirements.txt`.

## 2. Primary result reproduction

Run:

```bash
python src/reproduce_main_results.py
```

The script performs, in order:

1. Merge the included annotation audit with the 750 ms bipolar feature matrix.
2. Keep consensus AVP (`A`) and Physiological (`P`) recordings for the primary binary task.
3. Run nine outer leave-one-patient-out folds.
4. Within each outer training set, compare four LightGBM configurations using patient-held-out inner folds.
5. Fit Platt calibration only from inner out-of-fold probabilities.
6. Evaluate raw and calibrated probabilities on the untouched outer patient.
7. Compute pooled and patient-macro metrics.
8. Compute 10,000 patient-cluster bootstrap replicates for 95% intervals.
9. Learn selective-prediction thresholds from training-patient out-of-fold confidence only, then apply them to the outer patient.

Expected frozen values are available in `results/`.

## 3. Raw-data feature extraction

Download ARGO v1.0.0 from PhysioNet. Then run `src/extract_750ms_features.py`. The script reads the first channel (bipolar EGM), converts integer values using the header gain, keeps the final 750 samples (750 ms at 1 kHz), and calculates the 43 primary features.

## 4. Annotation audit

`notebooks/ARGO_Phase1_Audit.ipynb` documents the annotation parsing and disagreement analysis used to produce `data/audit/annotation_audit.csv`.

## 5. Temporal sensitivity and provenance

The exact original phase-4 scripts are preserved in `src/archive/`. Some retain the absolute paths used in the analysis environment and are included for provenance rather than portability. The cleaned scripts in `src/` are the recommended entry points.

## 6. Numerical reproducibility

Tree-based libraries can show tiny differences across LightGBM/compiler versions. The study's frozen outputs are included under `results/`, and the environment specification records the intended dependency ranges.
