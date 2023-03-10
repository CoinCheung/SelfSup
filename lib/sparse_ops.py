
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_add(out, identity):
    if isinstance(out, torch.Tensor): return out + identity
    return (e1 + e2 for e1,e2 in zip(out, identity))


class SparseConv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=True, padding_mode='zeros',
            device=None, dtype=None):
        super(SparseConv2d, self).__init__(
            in_chan, out_chan, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, device=device, dtype=dtype)

    def forward(self, x):
        '''
            x: nchw Tensor, or tuple/list of [x(nchw), mask(n1hw)]
        '''
        if isinstance(x, torch.Tensor): return super().forward(x)
        x, mask = x
        mask_ = mask.expand_as(x).detach()
        x = x.clone()
        x[~mask_] = 0.
        out = super().forward(x)
        sh, sw = self.stride
        mask = mask[:, :, ::sh, ::sw]
        #  mask_ = mask.expand_as(out)
        #  out[~mask_] = 0.
        return out, mask


class SparseBatchNorm2d(nn.BatchNorm1d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True, device=None, dtype=None):
        super(SparseBatchNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats, device=device,
            dtype=dtype)

    def forward(self, x):
        '''
            x: nchw Tensor, or tuple/list of [x(nchw), mask(n1hw)]
        '''

        if isinstance(x, torch.Tensor):
            n, c, h, w = x.size()
            x = torch.einsum('nchw->nhwc', x).reshape(-1, c)
            out = super().forward(x)
            out = out.reshape(n, h, w, c)
            out = torch.einsum('nhwc->nchw', out)
            return out

        x, mask = x
        n, c, h, w = x.size()
        x = torch.einsum('nchw->nhwc', x).reshape(-1, c)
        mask_ = torch.einsum('nchw->nhwc', mask).flatten()
        out = torch.zeros_like(x)
        out[mask_] = super().forward(x[mask_])
        out = out.reshape(n, h, w, c)
        out = torch.einsum('nhwc->nchw', out)
        return out, mask



class SparseMaxPool2d(nn.MaxPool2d):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(SparseMaxPool2d, self).__init__(
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        '''
            x: nchw Tensor, or tuple/list of [x(nchw), mask(n1hw)]
        '''
        if isinstance(x, torch.Tensor): return super().forward(x)
        x, mask = x
        mask_ = mask.expand_as(x).detach()
        x = x.clone()
        x[~mask_] = -torch.inf
        out = super().forward(x)
        sh, sw = self.stride, self.stride
        if isinstance(self.stride, (tuple, list)): sh, sw = self.stride
        mask = mask[:, :, ::sh, ::sw]
        #  mask_ = mask.expand_as(out)
        #  out[~mask_] = 0.
        return out, mask


class SparseReLU(nn.ReLU):

    def __init__(self, inplace=False):
        super(SparseReLU, self).__init__(inplace=inplace)

    def forward(self, x):
        '''
            x: nchw Tensor, or tuple/list of [x(nchw), mask(n1hw)]
        '''
        if isinstance(x, torch.Tensor): return super().forward(x)
        x, mask = x
        out = super().forward(x)
        #  mask_ = mask.expand_as(out)
        #  out[~mask_] = 0.
        return out, mask


if __name__ == '__main__':
    conv = SparseConv2d(3, 32, 3, 2, 1)
    bn = SparseBatchNorm2d(32)
    maxpool = SparseMaxPool2d(3, 2, 1)
    inten = torch.randn(2, 3, 224, 224)
    mask = torch.randint(0, 1, (2, 1, 224, 224)).bool()

    out1 = conv((inten, mask))[0]
    out1 = bn(out1)
    print(out1.size())
    out2 = conv(inten)
    out2 = bn(out2)
    print(out2.size())

    out3 = maxpool(inten)
    print(out3.size())

