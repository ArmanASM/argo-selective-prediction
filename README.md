# ARGO Selective Prediction Reproducibility Repository

Reproducibility materials for the manuscript:

**Learning When to Abstain: Expert Disagreement and Uncertainty-Aware Classification of Abnormal Ventricular Potentials in Post-Ischemic Ventricular Tachycardia**

This repository contains the analysis code, ARGO-derived feature matrix, annotation audit, final model outputs, and manuscript figures used in the study. **Raw ARGO waveform files are not redistributed here.**

## Study at a glance

The primary analysis uses a 750 ms bipolar intracardiac electrogram (EGM) window and 43 engineered signal features. A LightGBM classifier was evaluated with nested leave-one-patient-out (LOPO) validation across nine patients. The final analysis also evaluates probability calibration, patient-cluster bootstrap confidence intervals, and selective prediction using abstention thresholds learned only from training-patient out-of-fold predictions.

Headline validated results:

- Pooled accuracy: **93.24%**
- Pooled AUROC: **0.9720**
- Mean patient-balanced accuracy: **93.34%**
- At an 80% target coverage, realized held-out coverage was **78.15%** and retained-case accuracy was **97.39%**
- Abstained recordings were enriched for cases with initial expert disagreement

The frozen manuscript results are stored in `results/FINAL_methodology_results.csv`.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE-CODE
├── LICENSE-DATA.md
├── CITATION.cff.template
├── .gitignore
├── data/
│   ├── audit/annotation_audit.csv
│   └── derived/bipolar_last750ms_features.csv
├── src/
│   ├── reproduce_main_results.py
│   ├── extract_750ms_features.py
│   ├── feature_importance.py
│   └── archive/                 # original phase-4 analysis scripts
├── notebooks/
│   └── ARGO_Phase1_Audit.ipynb
├── results/                     # frozen outputs reported in the manuscript
├── figures/                     # final manuscript figures
└── docs/
    ├── REPRODUCIBILITY.md
    └── DATA_DICTIONARY.md
```

## Quick reproduction (no raw waveform download required)

The repository includes the derived 750 ms feature matrix and annotation audit used by the final model. This permits reproduction of the primary nested-LOPO and selective-prediction analysis without redistributing the raw ARGO waveform archive.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/smoke_test.py
python src/reproduce_main_results.py
```

Outputs will be written to `reproduced_results/`.

To recalculate held-out-patient permutation importance after the main run:

```bash
python src/feature_importance.py
```

## Full raw-data reproduction

The ARGO v1.0.0 source dataset is hosted by PhysioNet and is **not included in this repository**.

Dataset page: https://physionet.org/content/argo/1.0.0/

Version DOI: https://doi.org/10.13026/8gh2-e660

After downloading the ARGO ZIP, the 750 ms bipolar feature matrix can be regenerated with:

```bash
python src/extract_750ms_features.py \
  --argo-zip /path/to/annotated-dataset-of-post-ischemic-ventricular-tachycardia-electrograms-argo-1.0.0.zip \
  --output data/derived/bipolar_last750ms_features.csv
```

The Phase-1 notebook documents the multi-expert annotation audit. Original frozen phase-4 scripts used for the manuscript are preserved in `src/archive/` for provenance.

## Source data citation

If using ARGO, please cite both the PhysioNet dataset and the associated descriptor paper:

- Orrù M, Baldazzi G, Zirolia D, Bertagnolli L, Viola G, Solinas MG, Pani D. **Annotated dataset of post-ischemic ventricular tachycardia electrograms (ARGO)**, version 1.0.0. PhysioNet. 2026. doi:10.13026/8gh2-e660.
- Orrù M, Baldazzi G, Zirolia D, Bertagnolli L, Viola G, Solinas MG, Pani D. **The ARGO dataset: annotated and delineated intracardiac electrograms of post-ischemic ventricular tachycardia.** PLOS ONE. 2026;21(6):e0350993. doi:10.1371/journal.pone.0350993.

## Reproducibility notes

- Patient identity (`Pt1`–`Pt9`) is the grouping unit for all outer validation folds.
- No record from the held-out patient is used for model fitting, model selection, or selective-prediction threshold selection.
- Hyperparameter selection is performed inside each outer training set using patient-held-out inner folds.
- The primary task excludes consensus `Unknown` recordings and classifies consensus AVP (`A`) versus Physiological (`P`).
- Random seeds used in the frozen analysis are retained in the scripts.
- The repository includes frozen result CSVs so the reported manuscript values remain auditable even if future library versions produce minor numerical differences.

## Licenses

**Code:** MIT License; see `LICENSE-CODE`.

**ARGO-derived data and outputs:** The source ARGO files are licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. The ARGO-derived files in `data/`, `results/`, and `figures/` are provided under the same terms; see `LICENSE-DATA.md`.

## Citation of this repository

Before creating a Zenodo release, edit `CITATION.cff.template` with the actual author name(s), rename it to `CITATION.cff`, commit it, then create a GitHub release. Zenodo can archive that release and mint a DOI.

## Important limitation

ARGO contains 1,962 recordings but only nine patients. The repository reproduces a computational methodology study; these results should not be interpreted as independent clinical validation or evidence of clinical readiness.
