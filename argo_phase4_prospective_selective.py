from pathlib import Path
import numpy as np,pandas as pd,warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
from scipy import stats
warnings.filterwarnings('ignore')
OUT=Path('/mnt/data/argo_phase4_split')
a=pd.read_csv('/mnt/data/argo_phase1_results/annotation_audit.csv');f=pd.read_csv('/mnt/data/argo_phase2_initial/bipolar_last750ms_features.csv');fold=pd.read_csv(OUT/'nested_lopo_fold_metrics.csv');fold=fold[fold.version=='raw'];pred=pd.read_csv(OUT/'nested_lopo_predictions.csv')
d=a.merge(f,on=['patient','record']);d=d[d.consensus_label.isin(['A','P'])].reset_index(drop=True);cols=[c for c in f.columns if c not in ['patient','record']];X=d[cols].values;y=(d.consensus_label=='A').astype(int).values;pts=d.patient.values
cfg={'L7_m20':(7,20),'L15_m20':(15,20),'L31_m20':(31,20),'L15_m40':(15,40)};coverages=[.9,.8,.7,.6,.5];rows=[]
for op in sorted(d.patient.unique(),key=lambda s:int(s[2:])):
 name=fold.loc[fold.patient==op,'selected_config'].iloc[0];leaves,mc=cfg[name];tr=pts!=op;tridx=np.where(tr)[0];trainpts=[p for p in sorted(d.patient.unique(),key=lambda s:int(s[2:])) if p!=op]
 oof=np.full(tr.sum(),np.nan)
 for ip in trainpts:
  iva=pts[tridx]==ip;itr=~iva;m=LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=leaves,min_child_samples=mc,colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',random_state=42,verbosity=-1,n_jobs=-1);m.fit(X[tridx][itr],y[tridx][itr]);oof[iva]=m.predict_proba(X[tridx][iva])[:,1]
 trainconf=np.maximum(oof,1-oof)
 g=pred[pred.patient==op].copy().merge(a[['patient','record','n_unique']],on=['patient','record'],how='left');g['confidence']=np.maximum(g.pA_raw,1-g.pA_raw)
 fullacc=accuracy_score(g.y,g.pred_raw)
 for cov in coverages:
  threshold=float(np.quantile(trainconf,1-cov,method='linear'));keep=g[g.confidence>=threshold];ab=g[g.confidence<threshold]
  if len(keep)>0:
   ba=balanced_accuracy_score(keep.y,keep.pred_raw) if keep.y.nunique()>1 else np.nan;auc=roc_auc_score(keep.y,keep.pA_raw) if keep.y.nunique()>1 else np.nan;acc=accuracy_score(keep.y,keep.pred_raw)
  else: ba=auc=acc=np.nan
  rows.append({'patient':op,'target_coverage':cov,'threshold_from_training_oof':threshold,'actual_test_coverage':len(keep)/len(g),'retained_n':len(keep),'abstained_n':len(ab),'full_accuracy':fullacc,'retained_accuracy':acc,'accuracy_gain':acc-fullacc if len(keep)>0 else np.nan,'retained_balanced_accuracy':ba,'retained_auroc':auc,'retained_disagreement':(keep.n_unique>1).mean() if len(keep) else np.nan,'abstained_disagreement':(ab.n_unique>1).mean() if len(ab) else np.nan,'disagreement_enrichment':(ab.n_unique>1).mean()-(keep.n_unique>1).mean() if len(ab) and len(keep) else np.nan})
 print('done',op,flush=True)
r=pd.DataFrame(rows);r.to_csv(OUT/'prospective_selective_by_patient.csv',index=False)
# summarize pooled by applying each patient's training-derived threshold
summ=[];stat=[];rng=np.random.default_rng(42)
for cov in coverages:
 q=r[r.target_coverage==cov]
 # rebuild retained rows for pooled metrics
 kept=[]
 for _,rr in q.iterrows():
  g=pred[pred.patient==rr.patient].copy();g['confidence']=np.maximum(g.pA_raw,1-g.pA_raw);kept.append(g[g.confidence>=rr.threshold_from_training_oof])
 k=pd.concat(kept,ignore_index=True);yy=k.y.to_numpy();pr=k.pA_raw.to_numpy();pp=k.pred_raw.to_numpy()
 summ.append({'target_coverage':cov,'actual_pooled_coverage':len(k)/len(pred),'retained_n':len(k),'pooled_accuracy':accuracy_score(yy,pp),'pooled_balanced_accuracy':balanced_accuracy_score(yy,pp),'pooled_auroc':roc_auc_score(yy,pr),'mean_patient_accuracy_gain':q.accuracy_gain.mean(),'mean_disagreement_enrichment':q.disagreement_enrichment.mean()})
 for metric in ['accuracy_gain','disagreement_enrichment']:
  vals=q[metric].dropna().values
  try:w=stats.wilcoxon(vals,alternative='greater',method='exact')
  except:w=stats.wilcoxon(vals,alternative='greater')
  bb=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(10000)])
  stat.append({'target_coverage':cov,'quantity':metric,'mean':vals.mean(),'median':np.median(vals),'one_sided_wilcoxon_p':w.pvalue,'boot95_low':np.quantile(bb,.025),'boot95_high':np.quantile(bb,.975)})
pd.DataFrame(summ).to_csv(OUT/'prospective_selective_summary.csv',index=False);pd.DataFrame(stat).to_csv(OUT/'prospective_selective_stats.csv',index=False)
print(pd.DataFrame(summ).to_string(index=False));print(pd.DataFrame(stat).to_string(index=False))
