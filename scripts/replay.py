"""Causal magnetic-only replay of one released canonical Trial."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from csml import ReliableXYZ, features, effective_reset, output_contract
p=argparse.ArgumentParser();p.add_argument('--model',choices=['shared','holdout','static','no_slow','no_increment'],required=True);p.add_argument('--trial',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
rows=json.loads((ROOT/'data/index.json').read_text())['records'];row=next(x for x in rows if x['trial']==args.trial)
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
d=load_npz(ROOT/row['npz'])
if row.get('labels'):d.update(load_npz(ROOT/row['labels']))
d['n']=len(d['b_delta_uT']);norm=json.loads((ROOT/f'models/{args.model}_normalization.json').read_text());x=features(d,norm);reset=effective_reset(d)
if args.model in ['shared','holdout']:model=ReliableXYZ()
else:
 from csml.ablations import Ablated
 model=Ablated({'static':'S_static','no_slow':'F_no_slow','no_increment':'I_no_increment'}[args.model])
model=model.to(args.device);model.load_state_dict(torch.load(ROOT/f'models/{args.model}.pt',map_location=args.device,weights_only=True),strict=True);model.eval()
pred={'xyz_mm':np.full((d['n'],3),np.nan,np.float32),**{k:np.full(d['n'],np.nan,np.float32) for k in ['contact_probability','support_probability','expected_error_mm']}}
fast=slow=None;i=0
with torch.no_grad():
 while i<d['n']:
  if not d['input_valid'][i]:fast=slow=None;i+=1;continue
  if reset[i]:fast=slow=None
  end=i+1
  while end<d['n'] and d['input_valid'][end] and not reset[end] and end-i<127:end+=1
  tx=torch.as_tensor(x[i:end],device=args.device)[None];dt=torch.as_tensor(d['dt_s'][i:end],dtype=torch.float32,device=args.device)[None];active=torch.ones_like(dt,dtype=torch.bool)
  out=model(tx,dt,active,fast,slow)
  vals={'xyz_mm':out['xyz_normalized']*model.scale_mm,'contact_probability':torch.sigmoid(out['contact_logit']),'support_probability':torch.sigmoid(out['support_logit']),'expected_error_mm':out['expected_error_mm']}
  for k,v in vals.items():pred[k][i:end]=v[0].cpu().numpy()
  fast=out['fast_sequence'][:,-1][None];slow=out['slow_sequence'][:,-1];i=end
result={**pred,**output_contract(pred,d['input_valid']), 'input_valid':d['input_valid'],'t_group_ns':d['t_group_ns']}
np.savez_compressed(args.output,**result);print(args.output)
