import types
from .model import *
def no_slow_forward(m,x,dt,active,fast=None,slow=None):
 f,_=m.fast(x,fast);s=x.new_zeros(x.shape[0],x.shape[1],m.widths['slow'])
 mem=torch.cat([f,s],-1);stat=m.static(x);dyn=torch.cat([m.xy_dynamic(mem),m.depth_dynamic(mem)],-1)
 return dict(xyz_normalized=stat+dyn,static_normalized=stat,dynamic_normalized=dyn,contact_logit=m.contact(mem).squeeze(-1),fast_sequence=f,slow_sequence=s)

class Ablated(ReliableXYZ):
 def __init__(self, variant):
  self.variant=variant
  super().__init__()
  if self.variant=='S_static':
   s=self.widths['slow'];self.xyz.contact=nn.Sequential(nn.Linear(37,s),nn.SiLU(),nn.Linear(s,1))
   self.quality=nn.Sequential(nn.Linear(44,64),nn.SiLU(),nn.Linear(64,2))
   for key in ['fast','slow_drive','tau_logit','xy_dynamic','depth_dynamic']:delattr(self.xyz,key)
  elif self.variant=='F_no_slow':self.xyz.forward=types.MethodType(no_slow_forward,self.xyz)
 def forward(self,x,dt,active,fast=None,slow=None):
  if self.variant!='S_static':return super().forward(x,dt,active,fast,slow)
  q=self.xyz.static(x);dyn=torch.zeros_like(q);cp=self.xyz.contact(x).squeeze(-1)
  v=self.quality(torch.cat([x,q.detach(),dyn,torch.sigmoid(cp).detach()[...,None]],-1))
  el=F.softplus(v[...,1])
  return dict(xyz_normalized=q,static_normalized=q,dynamic_normalized=dyn,contact_logit=cp,support_logit=v[...,0],
   error_log1p=el,expected_error_mm=.5*torch.expm1(torch.clamp(el,max=12)),
   fast_sequence=x.new_zeros(x.shape[0],x.shape[1],self.widths['fast']),slow_sequence=x.new_zeros(x.shape[0],x.shape[1],self.widths['slow']))