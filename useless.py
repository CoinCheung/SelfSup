
import torch


pth = 'checkpoint_0199.pth.tar'


ckpt = torch.load(pth, map_location='cpu')
#  print(ckpt.keys())

state = ckpt['state_dict']
#  print(state.keys())

for k,v in state.items():
    if k.startswith('module.encoder_q.backbone.layer4'): print(k, ': ', v.size())


pth = 'pretrained/r50_checkpoint_0199_mocov2_cutmix.pth.tar'

ckpt = torch.load(pth, map_location='cpu')
#  print(ckpt.keys())

state = ckpt['state_dict']
#  print(state.keys())

print()
print()
for k,v in state.items():
    if k.startswith('module.encoder_q.layer4'): print(k, ': ', v.size())
