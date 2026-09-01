#!/usr/bin/env python3
"""Held-out-patient permutation importance for the nested-LOPO selected models."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, make_scorer

CFG = {'L7_m20': (7,20), 'L15_m20': (15,20), 'L31_m20': (31,20), 'L15_m40': (15,40)}

def make_model(leaves, min_child):
    return LGBMClassifier(n_estimators=180, learning_rate=.04, num_leaves=leaves,
        min_child_samples=min_child, colsample_bytree=.8, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbosity=-1, n_jobs=-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, default=None)
    args=ap.parse_args(); root=args.repo_root.resolve(); out=(args.output or root/'reproduced_results').resolve(); out.mkdir(parents=True, exist_ok=True)
    audit=pd.read_csv(root/'data/audit/annotation_audit.csv'); feat=pd.read_csv(root/'data/derived/bipolar_last750ms_features.csv')
    folds_path=out/'nested_lopo_fold_metrics.csv'
    if not folds_path.exists(): folds_path=root/'results/nested_lopo_fold_metrics.csv'
    fold=pd.read_csv(folds_path); fold=fold[fold.version=='raw']
    d=audit.merge(feat,on=['patient','record']); d=d[d.consensus_label.isin(['A','P'])].reset_index(drop=True)
    cols=[c for c in feat.columns if c not in ['patient','record']]; X=d[cols].values; y=(d.consensus_label=='A').astype(int).values; pts=d.patient.values
    rows=[]
    scorer=make_scorer(balanced_accuracy_score)
    for p in sorted(d.patient.unique(),key=lambda s:int(s[2:])):
        name=fold.loc[fold.patient==p,'selected_config'].iloc[0]; leaves,mc=CFG[name]; te=pts==p; tr=~te
        m=make_model(leaves,mc); m.fit(X[tr],y[tr])
        pi=permutation_importance(m,X[te],y[te],scoring=scorer,n_repeats=20,random_state=42,n_jobs=-1)
        gain=m.booster_.feature_importance(importance_type='gain').astype(float); gain=gain/(gain.sum() if gain.sum()>0 else 1)
        order=np.argsort(-pi.importances_mean); rank=np.empty_like(order); rank[order]=np.arange(1,len(order)+1)
        for j,c in enumerate(cols): rows.append({'patient':p,'feature':c,'perm_mean':pi.importances_mean[j],'perm_std':pi.importances_std[j],'perm_rank':int(rank[j]),'gain_norm':gain[j]})
        print('Completed',p)
    r=pd.DataFrame(rows); r.to_csv(out/'feature_importance_by_patient.csv',index=False)
    s=r.groupby('feature').agg(mean_perm=('perm_mean','mean'),median_perm=('perm_mean','median'),positive_fraction=('perm_mean',lambda x:(x>0).mean()),top10_fraction=('perm_rank',lambda x:(x<=10).mean()),mean_gain=('gain_norm','mean')).sort_values('mean_perm',ascending=False)
    s.to_csv(out/'feature_importance_stability.csv')
    print(s.head(15))
if __name__=='__main__': main()
