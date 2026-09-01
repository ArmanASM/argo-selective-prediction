from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,brier_score_loss
OUT=Path('/mnt/data/argo_phase4_split')
audit=pd.read_csv('/mnt/data/argo_phase1_results/annotation_audit.csv')
patients=[f'Pt{i}' for i in range(1,10)]
fold=pd.concat([pd.read_csv(OUT/f'fold_{p}.csv') for p in patients],ignore_index=True)
pred=pd.concat([pd.read_csv(OUT/f'pred_{p}.csv') for p in patients],ignore_index=True)
tune=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('tune_*.csv'))],ignore_index=True)
fold.to_csv(OUT/'nested_lopo_fold_metrics.csv',index=False);pred.to_csv(OUT/'nested_lopo_predictions.csv',index=False);tune.to_csv(OUT/'nested_tuning_scores.csv',index=False)

def ece(y,p,n_bins=10):
 e=0.;edges=np.linspace(0,1,n_bins+1)
 for i in range(n_bins):
  m=(p>=edges[i])&((p<edges[i+1]) if i<n_bins-1 else (p<=edges[i+1]))
  if m.any(): e += m.mean()*abs(y[m].mean()-p[m].mean())
 return e
rows=[]
for v in ['raw','platt']:
 pr=pred[f'pA_{v}'].to_numpy();y=pred.y.to_numpy();pp=pr>=.5; f=fold[fold.version==v]
 rows.append({'version':v,'pooled_accuracy':accuracy_score(y,pp),'pooled_balanced_accuracy':balanced_accuracy_score(y,pp),'pooled_f1':f1_score(y,pp),'pooled_auroc':roc_auc_score(y,pr),'brier':brier_score_loss(y,pr),'ece10':ece(y,pr),'macro_patient_accuracy':f.accuracy.mean(),'macro_patient_balanced_accuracy':f.balanced_accuracy.mean(),'macro_patient_auroc':f.auroc.mean(),'macro_patient_brier':f.brier.mean(),'macro_patient_ece10':f.ece10.mean()})
summary=pd.DataFrame(rows);summary.to_csv(OUT/'nested_calibration_summary.csv',index=False)
# bootstrap raw at patient cluster level
rng=np.random.default_rng(20260827);groups={p:pred[pred.patient==p] for p in patients};fraw=fold[fold.version=='raw'].set_index('patient')
boot=[]
for _ in range(10000):
 samp=rng.choice(patients,9,replace=True);g=pd.concat([groups[p] for p in samp],ignore_index=True);y=g.y.to_numpy();pr=g.pA_raw.to_numpy();pp=pr>=.5
 boot.append([accuracy_score(y,pp),balanced_accuracy_score(y,pp),roc_auc_score(y,pr),np.mean([fraw.loc[p,'balanced_accuracy'] for p in samp])])
boot=np.array(boot);raw=summary[summary.version=='raw'].iloc[0]
metricmap=[('pooled_accuracy',raw.pooled_accuracy),('pooled_balanced_accuracy',raw.pooled_balanced_accuracy),('pooled_auroc',raw.pooled_auroc),('macro_patient_balanced_accuracy',raw.macro_patient_balanced_accuracy)]
ci=[]
for j,(m,point) in enumerate(metricmap):ci.append({'metric':m,'point':point,'cluster_boot_95_low':np.quantile(boot[:,j],.025),'cluster_boot_95_high':np.quantile(boot[:,j],.975)})
pd.DataFrame(ci).to_csv(OUT/'nested_cluster_bootstrap_ci.csv',index=False)
# calibration paired stats at patient level
calstats=[]
for metric in ['brier','ece10','accuracy','balanced_accuracy']:
 r=fold[fold.version=='raw'].set_index('patient')[metric];c=fold[fold.version=='platt'].set_index('patient')[metric];d=(c-r).values
 try:w=stats.wilcoxon(d,method='exact')
 except:w=stats.wilcoxon(d)
 calstats.append({'metric':metric,'mean_platt_minus_raw':d.mean(),'median_delta':np.median(d),'wilcoxon_two_sided_p':w.pvalue})
pd.DataFrame(calstats).to_csv(OUT/'calibration_patient_stats.csv',index=False)
# nested selective prediction using raw (preserves uncalibrated ranking)
pred['confidence']=np.maximum(pred.pA_raw,1-pred.pA_raw)
sel=[]
for cov in [1,.9,.8,.7,.6,.5]:
 n=int(np.floor(cov*len(pred)));g=pred.nlargest(n,'confidence');y=g.y.to_numpy();pr=g.pA_raw.to_numpy();pp=pr>=.5
 sel.append({'coverage':cov,'n':n,'accuracy':accuracy_score(y,pp),'balanced_accuracy':balanced_accuracy_score(y,pp),'auroc':roc_auc_score(y,pr)})
pd.DataFrame(sel).to_csv(OUT/'nested_selective_global.csv',index=False)
# patientwise 80% and disagreement enrichment
g=pred.merge(audit[['patient','record','n_unique']],on=['patient','record'],how='left');ps=[]
for p,x in g.groupby('patient'):
 x=x.sort_values('confidence',ascending=False);n=int(np.floor(.8*len(x)));keep=x.iloc[:n];ab=x.iloc[n:]
 ps.append({'patient':p,'full_accuracy':accuracy_score(x.y,x.pred_raw),'retained_accuracy':accuracy_score(keep.y,keep.pred_raw),'accuracy_gain':accuracy_score(keep.y,keep.pred_raw)-accuracy_score(x.y,x.pred_raw),'retained_disagreement':(keep.n_unique>1).mean(),'abstained_disagreement':(ab.n_unique>1).mean(),'disagreement_enrichment':(ab.n_unique>1).mean()-(keep.n_unique>1).mean()})
ps=pd.DataFrame(ps);ps.to_csv(OUT/'nested_selective_80_patientwise.csv',index=False)
st=[]
for m in ['accuracy_gain','disagreement_enrichment']:
 d=ps[m].values
 try:w=stats.wilcoxon(d,alternative='greater',method='exact')
 except:w=stats.wilcoxon(d,alternative='greater')
 bb=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
 st.append({'quantity':m,'mean':d.mean(),'median':np.median(d),'wilcoxon_one_sided_p':w.pvalue,'boot95_low':np.quantile(bb,.025),'boot95_high':np.quantile(bb,.975)})
pd.DataFrame(st).to_csv(OUT/'nested_selective_patient_stats.csv',index=False)
# selected config frequency
freq=fold[fold.version=='raw'].selected_config.value_counts().rename_axis('config').reset_index(name='outer_folds_selected');freq.to_csv(OUT/'selected_config_frequency.csv',index=False)
print('SUMMARY\n',summary.to_string(index=False));print('\nCI\n',pd.DataFrame(ci).to_string(index=False));print('\nCALSTATS\n',pd.DataFrame(calstats).to_string(index=False));print('\nSELECTIVE\n',pd.DataFrame(sel).to_string(index=False));print('\nPATSTATS\n',pd.DataFrame(st).to_string(index=False));print('\nFREQ\n',freq.to_string(index=False))
