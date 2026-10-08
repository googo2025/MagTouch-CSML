"""Recompute paper statistics from fixed saved outputs, no neural inference."""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from csml.model import output_contract
p=argparse.ArgumentParser();p.add_argument('--protocol',choices=['shared','holdout','static','no_slow','no_increment'],required=True);a=p.parse_args()
index={x['trial']:x for x in json.loads((ROOT/'data/index.json').read_text())['records']};results=[]
for path in sorted((ROOT/'data/predictions'/a.protocol).glob('*.npz')):
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 mask=d['position_mask'].astype(bool);n=int(mask.sum())
 if not n:raise ValueError('Unexpected zero-target Trial in this manuscript population')
 e=d['xyz_mm'][mask].astype(float)-d['reference_xyz_mm'][mask].astype(float);assert np.isfinite(e).all()
 shift=e.mean(0);accepted=d.get('position_valid')
 if accepted is None:accepted=output_contract(d,d['input_valid'])['position_valid']
 def stats(v):
  q=np.linalg.norm(v,axis=1);return {'mean_mm':float(q.mean()),'p95_mm':float(np.percentile(q,95)),'rmse_mm':float(np.sqrt((q*q).mean()))}
 err=np.linalg.norm(e,axis=1);acc=accepted[mask].astype(bool);good=err<=.5
 results.append(dict(trial=path.stem,shape=index[path.stem]['shape'],targets=n,translation_xyz_mm=shift.tolist(),raw=stats(e),translation_removed=stats(e-shift),coverage=float(acc.mean()),bad_accepted_fraction=float((acc&~good).mean()),good_fraction=float(good.mean()),good_rejected_fraction=float((good&~acc).mean())))
shapes=sorted({r['shape'] for r in results});domains={}
for shape in shapes:
 group=[r for r in results if r['shape']==shape];entry={'trials':len(group),'targets':sum(r['targets'] for r in group)}
 for k in ['raw','translation_removed']:entry[k]={m:float(np.mean([r[k][m] for r in group])) for m in ['mean_mm','p95_mm','rmse_mm']}
 entry['coverage']=float(np.mean([r['coverage'] for r in group]));good=float(np.mean([r['good_fraction'] for r in group]));entry['accepted_bad_risk']=float(np.mean([r['bad_accepted_fraction'] for r in group])/entry['coverage']) if entry['coverage'] else None;entry['good_rejected']=float(np.mean([r['good_rejected_fraction'] for r in group])/good) if good else None;domains[shape]=entry
aggregate={k:{m:float(np.mean([d[k][m] for d in domains.values()])) for m in ['mean_mm','p95_mm','rmse_mm']} for k in ['raw','translation_removed']}
report=dict(protocol=a.protocol,trials=len(results),targets=sum(r['targets'] for r in results),domains=domains,aggregate=aggregate,aggregation='Equal eligible Trials within shape; equal shapes. P95/RMSE are means of Trial statistics.',translation_diagnostic_only=True)
expected={'shared':(97,132595,.2342655484),'holdout':(260,401803,.728708),'static':(260,401803,.9260232268),'no_slow':(260,401803,.730206),'no_increment':(260,401803,.780939)}[a.protocol];assert (report['trials'],report['targets'])==expected[:2];assert abs(aggregate['raw']['mean_mm']-expected[2])<1e-5
print(json.dumps(report,indent=2))
