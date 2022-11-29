
import pickle
import torch


pth = 'output_ckpt_200ep.pkl'
pth_cut = 'output_ckpt_200ep_cutmix.pkl'

with open(pth, 'rb') as fr:
    state = pickle.load(fr)['model']
with open(pth_cut, 'rb') as fr:
    state_cut = pickle.load(fr)['model']

#  state = torch.load(pth)
#  state_cut = torch.load(pth_cut)

for k,v in state.items():
    print(k, ": ", v.shape)


print()
print()
print()
for k,v in state_cut.items():
    print(k, ": ", v.shape)
