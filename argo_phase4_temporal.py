import zipfile,re,math,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats,signal
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score,accuracy_score,roc_auc_score
warnings.filterwarnings('ignore')
ZIP=Path('/mnt/data/annotated-dataset-of-post-ischemic-ventricular-tachycardia-electrograms-argo-1.0.0.zip')
ROOT='annotated-dataset-of-post-ischemic-ventricular-tachycardia-electrograms-argo-1.0.0/'
OUT=Path('/mnt/data/argo_phase4_split')
audit=pd.read_csv('/mnt/data/argo_phase1_results/annotation_audit.csv')
base=pd.read_csv('/mnt/data/argo_phase2_initial/bipolar_last750ms_features.csv')
records=base[['patient','record']].copy()
def safe_skew(x):
 v=stats.skew(x,bias=False); return 0.0 if not np.isfinite(v) else float(v)
def safe_kurt(x):
 v=stats.kurtosis(x,fisher=True,bias=False); return 0.0 if not np.isfinite(v) else float(v)
def autocorr(x,lag):
 if len(x)<=lag:return 0.0
 a=x[:-lag];b=x[lag:];sa=a.std();sb=b.std();return 0.0 if sa<1e-12 or sb<1e-12 else float(np.corrcoef(a,b)[0,1])
def bandpower(freq,power,lo,hi):
 m=(freq>=lo)&(freq<hi);return 0.0 if not m.any() else float(np.trapezoid(power[m],freq[m]))
def feats(x,fs=1000.):
 x=np.asarray(x,float);x=x-x.mean();dx=np.diff(x);absx=np.abs(x);eps=1e-12;maxa=float(absx.max()) if len(absx) else 0.
 pcs={}
 for frac in (.1,.2,.3,.5):
  pcs[frac]=0 if maxa<=eps else len(signal.find_peaks(absx,height=frac*maxa,distance=2)[0])
 f,p=signal.welch(x,fs=fs,nperseg=min(256,len(x)),noverlap=min(128,max(0,len(x)//2-1)),scaling='density')
 ps=p.sum();pn=p/ps if ps>eps else np.zeros_like(p);se=float(-(pn[pn>0]*np.log2(pn[pn>0])).sum()/np.log2(len(pn))) if (pn>0).sum()>1 else 0.
 cent=float((f*p).sum()/(p.sum()+eps));cdf=np.cumsum(p);med=float(f[np.searchsorted(cdf,cdf[-1]/2)]) if cdf[-1]>0 else 0.;dom=float(f[np.argmax(p)]) if len(p) else 0.
 b1=bandpower(f,p,16,40);b2=bandpower(f,p,40,80);b3=bandpower(f,p,80,125);b4=bandpower(f,p,125,250);b5=bandpower(f,p,250,500);lf=b1+b2;hf=b3+b4+b5
 te=x[1:-1]**2-x[:-2]*x[2:] if len(x)>=3 else np.array([0.]);varx=np.var(x);vard=np.var(dx) if len(dx) else 0.;mob=math.sqrt(vard/(varx+eps));dd=np.diff(dx);mobd=math.sqrt(np.var(dd)/(vard+eps)) if len(dd) else 0.;comp=mobd/(mob+eps)
 return {'mean':float(x.mean()),'std':float(x.std()),'rms':float(np.sqrt(np.mean(x*x))),'ptp':float(np.ptp(x)),'max_abs':maxa,'median_abs':float(np.median(absx)),'abs_q75':float(np.quantile(absx,.75)),'abs_q90':float(np.quantile(absx,.90)),'abs_q95':float(np.quantile(absx,.95)),'abs_q99':float(np.quantile(absx,.99)),'mad':float(np.median(np.abs(x-np.median(x)))),'skew':safe_skew(x),'kurtosis':safe_kurt(x),'line_length':float(np.abs(dx).sum()),'mean_abs_diff':float(np.mean(np.abs(dx))),'std_diff':float(np.std(dx)),'max_abs_diff':float(np.max(np.abs(dx))),'zero_cross_rate':float(np.mean(np.signbit(x[:-1])!=np.signbit(x[1:]))),'hjorth_mobility':mob,'hjorth_complexity':comp,'peaks_gt_0.1max':pcs[.1],'peaks_gt_0.2max':pcs[.2],'peaks_gt_0.3max':pcs[.3],'peaks_gt_0.5max':pcs[.5],'autocorr_1':autocorr(x,1),'autocorr_5':autocorr(x,5),'autocorr_10':autocorr(x,10),'autocorr_20':autocorr(x,20),'autocorr_50':autocorr(x,50),'autocorr_100':autocorr(x,100),'spectral_entropy':se,'spectral_centroid':cent,'median_freq':med,'dominant_freq':dom,'bandpower_16_40':b1,'bandpower_40_80':b2,'bandpower_80_125':b3,'bandpower_125_250':b4,'bandpower_250_500':b5,'hf_lf_ratio':float(hf/(lf+eps)),'veryhf_ratio':float(b5/(lf+hf+eps)),'teager_mean_abs':float(np.mean(np.abs(te))),'teager_std':float(np.std(te))}
def gain(hea):
 line=hea.strip().splitlines()[1].split();m=re.match(r'([0-9.+-eE]+)',line[2]);return float(m.group(1)) if m else 1.
windows=[250,500,1000,1500];rows={w:[] for w in windows}
with zipfile.ZipFile(ZIP) as z:
 for i,r in records.iterrows():
  pref=f'{ROOT}ARGODataset_Folder/{r.patient}/{r.record}';raw=np.frombuffer(z.read(pref+'.dat'),dtype='<i4').reshape(-1,15);g=gain(z.read(pref+'.hea').decode(errors='replace'));b=raw[:,0]/g
  for w in windows:
   d={'patient':r.patient,'record':r.record};d.update(feats(b[-w:]));rows[w].append(d)
  if (i+1)%300==0:print('extracted',i+1,flush=True)
for w in windows:pd.DataFrame(rows[w]).to_csv(OUT/f'bipolar_last{w}ms_features.csv',index=False)
print('feature extraction complete',flush=True)
allwins=windows+[750];res=[]
for w in allwins:
 f=base if w==750 else pd.read_csv(OUT/f'bipolar_last{w}ms_features.csv');d=audit.merge(f,on=['patient','record']);d=d[d.consensus_label.isin(['A','P'])].reset_index(drop=True);cols=[c for c in f.columns if c not in ['patient','record']];X=d[cols].values;y=(d.consensus_label=='A').astype(int).values;pts=d.patient.values
 for p in sorted(d.patient.unique(),key=lambda s:int(s[2:])):
  te=pts==p;tr=~te;m=LGBMClassifier(n_estimators=180,learning_rate=.04,num_leaves=31,min_child_samples=20,colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',random_state=42,verbosity=-1,n_jobs=-1);m.fit(X[tr],y[tr]);pr=m.predict_proba(X[te])[:,1]
  res.append({'window_ms':w,'patient':p,'accuracy':accuracy_score(y[te],pr>=.5),'balanced_accuracy':balanced_accuracy_score(y[te],pr>=.5),'auroc':roc_auc_score(y[te],pr)})
 print('modeled',w,flush=True)
r=pd.DataFrame(res);r.to_csv(OUT/'temporal_window_lopo_folds.csv',index=False);s=r.groupby('window_ms').agg(macro_accuracy=('accuracy','mean'),macro_balanced_accuracy=('balanced_accuracy','mean'),macro_auroc=('auroc','mean')).sort_index();s.to_csv(OUT/'temporal_window_summary.csv');print(s,flush=True)
