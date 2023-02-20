import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.sparse_ops import SparseBatchNorm2d


class MAEDecoder(nn.Module):

    def __init__(self, chans):
        super(MAEDecoder, self).__init__()
        self.chans = chans
        self.conv32 = self.get_decode_conv(chans[-1], chans[-2])
        self.conv16 = self.get_decode_conv(chans[-2], chans[-3])
        self.conv8 = self.get_decode_conv(chans[-3], chans[-4])
        self.conv4 = self.get_decode_conv(chans[-4], chans[-5])
        self.conv_proj = nn.Conv2d(chans[-5], 3, 1, 1, 0, bias=True)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = x
        feat = self.conv32(feat32)
        feat = feat + feat16
        feat = self.conv16(feat)
        feat = feat + feat8
        feat = self.conv8(feat)
        feat = feat + feat4
        feat = self.conv4(feat)
        feat = self.conv_proj(feat)
        return feat

    def get_decode_conv(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Upsample(scale_factor=2., mode='bilinear', align_corners=False),
            nn.Conv2d(in_chan, in_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
        )


class MAEMasker(nn.Module):

    def __init__(self, enc_chans, dec_chans, mask_ratio=0.6):

        super(MAEMasker, self).__init__()
        self.mask_ratio = mask_ratio
        assert len(enc_chans) == len(dec_chans)
        self.norms = nn.ModuleList()
        self.lin_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        self.psize = 32

        for i, (enc_ch, dec_ch) in enumerate(zip(enc_chans, dec_chans)):
            self.norms.append(SparseBatchNorm2d(enc_ch))

            if i == len(enc_chans) - 1:
                if enc_ch == dec_ch: proj = nn.Identity()
                else: proj = nn.Conv2d(enc_ch, dec_ch, 1, 1, 0, bias=True)
            else:
                proj = nn.Conv2d(enc_ch, dec_ch, 3, 1, 1, bias=True)
            self.lin_projs.append(proj)

            p = nn.Parameter(torch.empty(1, enc_ch, 1, 1))
            nn.init.trunc_normal_(p, std=.02)
            self.mask_tokens.append(p)


    @torch.no_grad()
    def gen_mask(self, x):
        N, _, H, W = x.size()
        device = x.device
        psize = self.psize
        nh, nw = H // psize, W // psize
        mask = torch.ones((N, H, W), device=device, dtype=torch.bool)
        mask = mask.view(N, nh, psize, nw, psize)
        mask = torch.einsum('nhpwq->nhwpq', mask).reshape(-1, psize * psize)
        nptchs = mask.size(0)
        nmsk = round(self.mask_ratio * nptchs)
        inds = torch.randperm(nptchs, device=device)[:nmsk]
        mask[inds, :] = 0
        mask = mask.reshape(N, nh, nw, psize, psize)
        mask = torch.einsum('nhwpq->nhpwq', mask).reshape(N, 1, H, W)
        return mask.detach()

    def replace_mask(self, feats):
        '''
            feats: ((feat4, mask4), (feat8, mask8), ...)
        '''
        res = []
        for i, (feat, mask) in enumerate(feats):
            feat = self.norms[i](feat)
            mtoks = self.mask_tokens[i].expand_as(feat)
            feat = torch.where(mask, feat, mtoks)
            feat = self.lin_projs[i](feat)
            res.append(feat)
        return res

    @torch.no_grad()
    def patch_norm_target(self, target):
        N, C, H, W = target.size()
        psize = self.psize
        nh, nw = H // psize, W // psize
        target = target.reshape(N, C, nh, psize, nw, psize)
        target = torch.einsum('nchpwq->nhwpqc', target)
        target = target.reshape(-1, psize * psize * C)
        var, mean = torch.var_mean(target, dim=1,
                unbiased=False, keepdim=True)
        eps = 1e-6
        rstd = (var + eps).rsqrt()
        target = (target - mean) * rstd
        target = target.reshape(N, nh, nw, psize, psize, C)
        target = torch.einsum('nhwpqc->nchpwq', target)
        target = target.reshape(N, C, H, W)
        return target.detach()

    def compute_l2_loss(self, feat, target, mask):
        target = self.patch_norm_target(target)
        loss = (feat - target) ** 2
        loss = loss.mean(dim=1, keepdim=True)
        loss = loss[~mask].mean()
        return loss


#  decoder = MAEDecoder([64, 128, 256, 512])
#
#  feat4 = torch.randn(5, 64, 56, 56)
#  feat8 = torch.randn(5, 128, 28, 28)
#  feat16 = torch.randn(5, 256, 14, 14)
#  feat32 = torch.randn(5, 512, 7, 7)
#  out = decoder((feat4, feat8, feat16, feat32))
#  print(out.size())
