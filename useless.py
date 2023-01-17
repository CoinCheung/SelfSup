
import torch

#
#  pth = 'checkpoint_0199.pth.tar'
#
#
#  ckpt = torch.load(pth, map_location='cpu')
#  #  print(ckpt.keys())
#
#  state = ckpt['state_dict']
#  #  print(state.keys())
#
#  for k,v in state.items():
#      if k.startswith('module.encoder_q.backbone.layer4'): print(k, ': ', v.size())
#
#
#  pth = 'pretrained/r50_checkpoint_0199_mocov2_cutmix.pth.tar'
#
#  ckpt = torch.load(pth, map_location='cpu')
#  #  print(ckpt.keys())
#
#  state = ckpt['state_dict']
#  #  print(state.keys())
#
#  print()
#  print()
#  for k,v in state.items():
#      if k.startswith('module.encoder_q.layer4'): print(k, ': ', v.size())




im = torch.randn(4, 3, 3, 3)
imflip = im.flip(dims=(2,))
probs = torch.rand(im.size(0))[:, None, None, None]
res1 = torch.where(probs > 0.5, im, imflip)
res2 = torch.empty_like(im)
for i in range(im.size(0)):
    if probs[i, 0, 0, 0] > 0.5:
        res2[i, ...] = im[i, ...]
    else:
        res2[i, ...] = imflip[i, ...]

print((res1 - res2).abs().max())
