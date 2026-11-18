"""Reproducible full-validation analysis; run from repository root."""
import json,copy,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aml.scaling import fit_power,predict,counts
b=json.loads(Path('report/complete_results.json').read_text());full=json.loads(Path('report/full_validation_all.json').read_text())
out=Path('report/final_analysis');out.mkdir(exist_ok=True)
rows=copy.deepcopy(b['rows'])
for r in rows:
 v=full[r['id']];assert v['tokens']==4958034 and v['bytes']==19004680 and v['max_documents']==0
 r['subset_loss']=r['loss'];r.update({k:v[k] for k in ['loss','ppl','bpb','tokens','bytes']})
f=sorted([r for r in rows if r['cohort']=='fixed_d'],key=lambda r:r['N']);main=full['main'];target=b['main']['N']
def logfit(rs):
 x=np.array([r['N']/1e7 for r in rs]);y=np.array([r['loss'] for r in rs]);opts=[]
 for e in [0,.5,1.]:
  z=least_squares(lambda p:np.log(p[0]+p[1]*x**(-p[2]))-np.log(y),[e,1,.3],bounds=([0,1e-9,.001],[min(y),100,3]),max_nfev=10000);opts.append(z)
 z=min(opts,key=lambda x:sum(x.fun**2));e,a,al=z.x;return {'E':float(e),'A':float(a*1e7**al),'alpha':float(al)}
fit=fit_power([r['N'] for r in f],[r['loss'] for r in f]);lf=logfit(f)
checks=[]
for name,fn in [('loss least squares',lambda rs:fit_power([r['N'] for r in rs],[r['loss'] for r in rs])),('log-loss least squares',logfit)]:
 for train,hold in [(f[:4],f[4]),(f,dict(N=target,loss=main['loss']))]:
  z=fn(train);p=predict(z,hold['N']);checks.append(dict(method=name,n_fit=len(train),target_N=hold['N'],predicted=p,measured=hold['loss'],relative_error=abs(p-hold['loss'])/hold['loss']))
sens=[predict(fit_power([r['N'] for j,r in enumerate(f) if j!=i],[r['loss'] for j,r in enumerate(f) if j!=i]),target) for i in range(5)]
front=[]
for c in [1e16,3e16,1e17]:
 group=sorted([r for r in rows if r['cohort']=='isoflop' and r['budget']==c],key=lambda r:r['N']);w=min(group,key=lambda r:r['loss']);front.append(dict(budget=c,N=w['N'],D=w['D'],loss=w['loss'],boundary=w==group[0] or w==group[-1]))
cf=fit_power([r['budget'] for r in front],[r['loss'] for r in front],offset=False)
summary=dict(fixed_fit=fit,log_fit=lf,checks=checks,leave_one_out=sens,frontier=front,compute_fit=cf,main=main,rows=rows)
(out/'analysis.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10})
fig,axs=plt.subplots(1,2,figsize=(10,3.7))
x=np.geomspace(f[0]['N'],target,200)
axs[0].plot(x/1e6,[predict(fit,v) for v in x],label='Fit to five small models');axs[0].scatter([r['N']/1e6 for r in f],[r['loss'] for r in f],label='Full-validation observations');axs[0].scatter(target/1e6,main['loss'],marker='*',s=110,label='Held-out main observation');axs[0].scatter(target/1e6,predict(fit,target),marker='x',s=80,label='Main prediction');axs[0].set(xscale='log',xlabel='Non-embedding parameters (M)',ylabel='Full-validation loss');axs[0].legend(fontsize=7)
for c in [1e16,3e16,1e17]:
 rr=sorted([r for r in rows if r['cohort']=='isoflop' and r['budget']==c],key=lambda r:r['N']);axs[1].plot([r['N']/1e6 for r in rr],[r['loss'] for r in rr],'-o',label=f'{c:.0e} FLOPs')
axs[1].set(xscale='log',xlabel='Non-embedding parameters (M)',ylabel='Full-validation loss');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'fits.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3));x=np.geomspace(1e16,1e17,100);ax.plot(x,[predict(cf,v) for v in x],'--',label='Zero-offset descriptive fit');ax.scatter([r['budget'] for r in front],[r['loss'] for r in front],label='Best sampled full-val loss');ax.set(xscale='log',xlabel='Approximate FLOPs',ylabel='Full-validation loss');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'frontier.png',dpi=200);plt.close(fig)
# Histories use the monitoring subset, not full-validation final evaluations.
curves=copy.deepcopy(b['curves']);pilot=json.loads(Path('experiments/path1_pilot/plan.json').read_text())
meta={r['id']:r for r in b['plan']['runs']}
for r in pilot['runs']:
 if r['cohort']=='fixed_d':
  meta[r['id']]=r;d=Path('report/evidence/course/runs/scaling')/r['id'];curves[r['id']]=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]
groups=[('Pilot fixed D',[k for k,r in meta.items() if r['cohort']=='fixed_d' and r['D']==67108864]),('Matched-main fixed D',[r['id'] for r in f])]+[(f'IsoFLOP {c:.0e}',[r['id'] for r in rows if r['cohort']=='isoflop' and r['budget']==c]) for c in [1e16,3e16,1e17]]
fig,axs=plt.subplots(5,2,figsize=(10,13))
for i,(name,ids) in enumerate(groups):
 for k in sorted(ids,key=lambda k:meta[k]['N']):
  r=meta[k];vs=[v for v in curves[k] if v['kind']=='val' and v['tokens_seen']>0]
  for j in [0,1]:axs[i,j].plot([v['tokens_seen']*(6*r['N'] if j else 1) for v in vs],[v['loss'] for v in vs],label=f"{r['N']/1e6:.2f}M")
 for j in [0,1]:
  axs[i,j].set(xscale='log',xlabel='Approx. FLOPs' if j else 'Processed tokens',ylabel='Subset loss',title=name);axs[i,j].legend(fontsize=6,ncol=3)
fig.tight_layout();fig.savefig(out/'all_30_histories.png',dpi=200);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
