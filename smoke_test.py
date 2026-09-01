"""Fast integrity test for repository inputs and frozen headline results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
feat = pd.read_csv(ROOT / 'data/derived/bipolar_last750ms_features.csv')
audit = pd.read_csv(ROOT / 'data/audit/annotation_audit.csv')
final = pd.read_csv(ROOT / 'results/FINAL_methodology_results.csv')

assert feat.shape == (1962, 45), f'Unexpected feature matrix shape: {feat.shape}'
assert len(feat.columns) - 2 == 43
assert audit.shape[0] == 1962
assert audit.consensus_label.value_counts().to_dict() == {'A': 940, 'P': 776, 'U': 246}
assert audit.patient.nunique() == 9

items = dict(zip(final['Item'], final['Final result']))
assert items['Pooled accuracy'] == '0.9324'
assert items['Pooled AUROC'] == '0.9720'
assert items['Macro patient-balanced accuracy'] == '0.9334'
print('ARGO repository smoke test: PASS')
