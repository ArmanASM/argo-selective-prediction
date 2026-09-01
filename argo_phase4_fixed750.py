from pathlib import Path
import pandas as pd,numpy as np,warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_auc_score
warnings.filterwarnings('ignore')
OUT=Path('/mnt/data/argo_phase4_split')
a=pd.read_csv('/mnt/data/argo_phase1_results/annotation_audit.csv');f=pd.read_csv('/mnt/data/argo_phase2_initial/bipolar_last750ms_features.csv')
d=a.merge(f,on=['patient','record']);d=d[d.consensus_label.isin(['A','P'])].reset_index(drop=True)
cols=[c for c in f.columns if c not in ['patient','record']];X=d[cols].values;y=(d.consensus_label=='A').astype(int).values;pts=d.patient.values
cfgs=[('L7_m20',7,20),('L15_m20',15,20),('L31_m20',31,20),('L15_m40',15,40)]
rows=[]
for name,leaves,mc in cfgs:
 for p in sorted(d.patient.unique(),key=lambda s:int(s[2:])):
  te=pts==p;tr=~te;m=LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=leaves,min_child_samples=mc,colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',random_state=42,verbosity=-1,n_jobs=-1);m.fit(X[tr],y[tr]);pr=m.predict_proba(X[te])[:,1]
  rows.append([name,p,balanced_accuracy_score(y[te],pr>=.5),accuracy_score(y[te],pr>=.5),roc_auc_score(y[te],pr)])
 print(name,flush=True)
r=pd.DataFrame(rows,columns=['config','patient','balanced_accuracy','accuracy','auroc']);r.to_csv(OUT/'fixed_config_750_folds.csv',index=False)
s=r.groupby('config').agg(macro_balanced_accuracy=('balanced_accuracy','mean'),macro_accuracy=('accuracy','mean'),macro_auroc=('auroc','mean')).sort_values('macro_balanced_accuracy',ascending=False);s.to_csv(OUT/'fixed_config_750_summary.csv')
print(s)
