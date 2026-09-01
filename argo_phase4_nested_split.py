from pathlib import Path
import sys, json, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
warnings.filterwarnings('ignore')
OUT=Path('/mnt/data/argo_phase4_split'); OUT.mkdir(exist_ok=True)
audit=pd.read_csv('/mnt/data/argo_phase1_results/annotation_audit.csv')
feat=pd.read_csv('/mnt/data/argo_phase2_initial/bipolar_last750ms_features.csv')
df=audit.merge(feat,on=['patient','record'],how='inner'); df=df[df.consensus_label.isin(['A','P'])].reset_index(drop=True)
cols=[c for c in feat.columns if c not in ('patient','record')]
X=df[cols].to_numpy(float); y=(df.consensus_label=='A').astype(int).to_numpy(); pts=df.patient.to_numpy(); rec=df.record.to_numpy()
patients=sorted(df.patient.unique(),key=lambda s:int(s[2:]))
PARAMS=[
 {'name':'L7_m20','num_leaves':7,'min_child_samples':20},
 {'name':'L15_m20','num_leaves':15,'min_child_samples':20},
 {'name':'L31_m20','num_leaves':31,'min_child_samples':20},
 {'name':'L15_m40','num_leaves':15,'min_child_samples':40},
]
def model(cfg):
 return LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=cfg['num_leaves'],min_child_samples=cfg['min_child_samples'],colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',random_state=42,verbosity=-1,n_jobs=-1)
def ece(y,p,n_bins=10):
 e=0.; edges=np.linspace(0,1,n_bins+1)
 for i in range(n_bins):
  m=(p>=edges[i]) & ((p<edges[i+1]) if i<n_bins-1 else (p<=edges[i+1]))
  if m.any(): e += m.mean()*abs(y[m].mean()-p[m].mean())
 return e
requested=sys.argv[1:] or patients
for outer_p in requested:
 te=pts==outer_p; tr=~te; tr_idx=np.where(tr)[0]; inner_pat=[p for p in patients if p!=outer_p]
 scores=[]; oofs={}
 for cfg in PARAMS:
  oof=np.full(tr.sum(),np.nan); bas=[]; aucs=[]
  for ip in inner_pat:
   iva=pts[tr_idx]==ip; itr=~iva; m=model(cfg); m.fit(X[tr_idx][itr],y[tr_idx][itr]); pr=m.predict_proba(X[tr_idx][iva])[:,1]; oof[iva]=pr
   yy=y[tr_idx][iva]; bas.append(balanced_accuracy_score(yy,pr>=.5)); aucs.append(roc_auc_score(yy,pr))
  scores.append((np.mean(bas),np.mean(aucs),cfg['name'])); oofs[cfg['name']]=oof
  pd.DataFrame([{'outer_patient':outer_p,'config':cfg['name'],'inner_macro_balanced_accuracy':np.mean(bas),'inner_macro_auroc':np.mean(aucs)}]).to_csv(OUT/f'tune_{outer_p}_{cfg["name"]}.csv',index=False)
 scores.sort(reverse=True); best_name=scores[0][2]; cfg=next(c for c in PARAMS if c['name']==best_name); oof=oofs[best_name]
 cal=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000); cal.fit(logit(np.clip(oof,1e-5,1-1e-5)).reshape(-1,1),y[tr_idx])
 m=model(cfg); m.fit(X[tr],y[tr]); raw=m.predict_proba(X[te])[:,1]; plc=cal.predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1]; yy=y[te]
 fold=[]
 for version,pr in [('raw',raw),('platt',plc)]:
  pp=pr>=.5; fold.append({'patient':outer_p,'version':version,'selected_config':best_name,'n':len(yy),'accuracy':accuracy_score(yy,pp),'balanced_accuracy':balanced_accuracy_score(yy,pp),'f1':f1_score(yy,pp),'auroc':roc_auc_score(yy,pr),'brier':brier_score_loss(yy,pr),'ece10':ece(yy,pr)})
 pd.DataFrame(fold).to_csv(OUT/f'fold_{outer_p}.csv',index=False)
 pred=pd.DataFrame({'patient':outer_p,'record':rec[te],'y':yy,'pA_raw':raw,'pA_platt':plc,'pred_raw':(raw>=.5).astype(int),'pred_platt':(plc>=.5).astype(int),'selected_config':best_name})
 pred.to_csv(OUT/f'pred_{outer_p}.csv',index=False)
 print(outer_p,best_name,'rawBA',fold[0]['balanced_accuracy'],'rawAUC',fold[0]['auroc'],'rawBrier',fold[0]['brier'],'plattBrier',fold[1]['brier'],flush=True)
