import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SCALE_MM=[10.,10.,3.]
BASE_PARAMETERS=44551

def effective_reset(d):
    good = d["input_valid"].astype(bool)
    reset = d["reset"].astype(bool).copy()
    reset[0] = True
    reset |= ~good
    reset[1:] |= ~good[:-1] | (d["dt_s"][1:] > 1.0)
    return reset

def features(d, norm):
    valid = d["input_valid"].astype(bool)
    mask = np.repeat(d["sensor_valid"].astype(bool), 3, axis=1) & valid[:, None]
    b = np.where(mask, d["b_delta_uT"], 0).astype(np.float32)
    amp = np.linalg.norm(b, axis=1, keepdims=True)
    raw = np.where(mask, (b - norm["b_mean_uT"]) / norm["b_std_uT"], 0)
    direction = b / np.maximum(amp, 1.)
    logamp = np.log(np.maximum(amp, 1.) / norm["amplitude_scale_uT"])
    dt = np.log1p(np.clip(d["dt_s"], 0, 1)[:, None] / .03)
    x = np.concatenate([raw, direction, logamp, dt, d["sensor_valid"].astype(float)], axis=1)
    x[~valid] = 0
    if x.shape != (d["n"], 37) or not np.isfinite(x).all():
        raise ValueError("Invalid magnetic-only features")
    return x.astype(np.float32)

def slow_scan(drive: torch.Tensor, dt: torch.Tensor, state: torch.Tensor,
              tau: torch.Tensor, active: torch.Tensor):
    # Exact linear recurrence within bounded blocks: no JIT graph compilation
    # or per-frame Python/kernel overhead. Float64 keeps exp(S) finite even
    # with the extreme dt=1s, tau=.1s: block 32 -> S<=320 < log(DBL_MAX).
    outputs = []
    carry = state.double()
    for start in range(0, drive.size(1), 32):
        end = min(drive.size(1), start+32)
        delta = torch.where(active[:, start:end], dt[:, start:end], 0.).double()
        delta = torch.clamp(delta, 0., 1.)[:, :, None] / tau.double()[None, None, :]
        cumulative = torch.cumsum(delta, dim=1)
        weighted = -torch.expm1(-delta) * drive[:, start:end].double() * torch.exp(cumulative)
        sequence = torch.exp(-cumulative) * (carry[:, None] + torch.cumsum(weighted, dim=1))
        carry = sequence[:, -1]
        outputs.append(sequence.to(drive.dtype))
    return torch.cat(outputs, dim=1)

def dimensions(factor):
    def widths(s):
        return {'static1':max(1,round(96*s)), 'static2':max(1,round(64*s)),
                'fast':max(1,round(64*s)), 'slow':max(1,round(32*s))}
    def count(w):
        a,b,h,s=w['static1'],w['static2'],w['fast'],w['slow']
        return (38*a+(a+1)*b+3*(b+1) + 111*h+3*h*h+6*h + (h+1)*s+s
                + (h+s+1)*b+2*(b+1) + 2*((h+s+1)*s+(s+1)))
    lo,hi=1.,max(1.,2*math.sqrt(factor))
    target=BASE_PARAMETERS*factor
    candidates=[]
    for _ in range(64):
        mid=(lo+hi)/2; w=widths(mid); n=count(w); candidates.append((abs(n-target),w))
        if n<target: lo=mid
        else: hi=mid
    return min(candidates,key=lambda item:item[0])[1]

class ScaledXYZ(nn.Module):
    def __init__(self, factor):
        super().__init__()
        w=dimensions(factor)
        a,b,h,s=w['static1'],w['static2'],w['fast'],w['slow']
        self.widths=w
        self.static=nn.Sequential(nn.Linear(37,a),nn.SiLU(),nn.Linear(a,b),nn.SiLU(),nn.Linear(b,3))
        self.fast=nn.GRU(37,h,batch_first=True)
        self.slow_drive=nn.Sequential(nn.Linear(h,s),nn.Tanh())
        self.tau_logit=nn.Parameter(torch.full((s,),math.log((.45-.1)/(2.-.45))))
        self.xy_dynamic=nn.Sequential(nn.Linear(h+s,b),nn.SiLU(),nn.Linear(b,2))
        self.depth_dynamic=nn.Sequential(nn.Linear(h+s,s),nn.SiLU(),nn.Linear(s,1))
        self.contact=nn.Sequential(nn.Linear(h+s,s),nn.SiLU(),nn.Linear(s,1))
        for head in [self.xy_dynamic,self.depth_dynamic]:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        self.register_buffer('scale_mm',torch.tensor(SCALE_MM))

    def forward(self,x,dt,active,fast=None,slow=None):
        f,_=self.fast(x,fast)
        if slow is None:
            slow=x.new_zeros(x.size(0),self.widths['slow'])
        tau=.1+1.9*torch.sigmoid(self.tau_logit)
        s=slow_scan(self.slow_drive(f),dt,slow,tau,active)
        memory=torch.cat([f,s],dim=-1)
        static=self.static(x)
        delta=torch.cat([self.xy_dynamic(memory),self.depth_dynamic(memory)],dim=-1)
        return {'xyz_normalized':static+delta,'static_normalized':static,'dynamic_normalized':delta,
                'contact_logit':self.contact(memory).squeeze(-1),'fast_sequence':f,'slow_sequence':s}

class ReliableXYZ(nn.Module):
    def __init__(self):
        super().__init__();self.xyz=ScaledXYZ(10)
        dim=self.xyz.widths['fast']+self.xyz.widths['slow']+44
        self.quality=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,2))
        self.widths=self.xyz.widths

    @property
    def scale_mm(self):return self.xyz.scale_mm

    def forward(self,x,dt,active,fast=None,slow=None):
        o=self.xyz(x,dt,active,fast,slow)
        memory=torch.cat([o['fast_sequence'],o['slow_sequence']],dim=-1)
        # Reliability cannot improve itself by perturbing XYZ or its memory.
        o['contact_logit']=self.xyz.contact(memory.detach()).squeeze(-1)
        extra=torch.cat([o['fast_sequence'].detach(),o['slow_sequence'].detach(),x,
            o['xyz_normalized'].detach(),o['dynamic_normalized'].detach(),
            torch.sigmoid(o['contact_logit']).detach()[...,None]],dim=-1)
        v=self.quality(extra);o['support_logit']=v[...,0]
        o['error_log1p']=F.softplus(v[...,1])
        o['expected_error_mm']=.5*torch.expm1(torch.clamp(o['error_log1p'],max=12))
        return o

def output_contract(prediction,input_valid):
    q=prediction['xyz_mm'];cp=prediction['contact_probability'];support=prediction['support_probability'];err=prediction['expected_error_mm']
    finite=np.isfinite(q).all(1)&np.isfinite(cp)&np.isfinite(support)&np.isfinite(err)&input_valid
    accepted=finite&(cp>=.5)&(support>=.5)&(err<=.5)
    state=np.full(len(q),-1,np.int8)
    state[finite&(cp<.5)]=0
    state[finite&(cp>=.5)]=1
    state[accepted]=2
    # No-contact has no tactile XY reference. Weak contact is not forced to d=0.
    published=np.where(accepted[:,None],q,np.nan)
    depth=np.where(state==0,0.,np.where(accepted,q[:,2],np.nan))
    return {'position_valid':accepted,'output_state':state,'published_xyz_mm':published,'published_indentation_mm':depth}
