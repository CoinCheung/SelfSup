
import torch

state = torch.load('output/pixpro_base_r50_100ep/ckpt_epoch_100.pth')['model']

new_state = {
        k.replace('module.encoder.backbone.', ''):v
        for k,v in state.items()
        if 'encoder.backbone' in k and not 'filt' in k}

print(new_state.keys())

new_ckpt = {'backbone': new_state}


torch.save(new_ckpt, 'model_final_r50_ibn_b_pixpro.pth')



