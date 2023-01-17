import random

from PIL import ImageFilter
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class NCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_crops=2):
        self.base_transform = base_transform
        self.n_crops = n_crops

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_crops)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ToTensorBatchGPU(object):
    '''
        input is im: nhwc tensor, uint8 (merely after augment)
        we first transpose it into nchw, then cast to float32 and divide 255. like to_tensor
        after that, we subtract mean and divide it by std
    '''
    def __init__(self):
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        self.mean = mean[None, :, None, None].detach()
        self.std = std[None, :, None, None].detach()

    @torch.no_grad()
    def __call__(self, im):
        im = self.random_hflip(im) # random hflip
        im = im.permute(0, 3, 1, 2).float().div_(255.) # to tensor
        im = im.sub_(self.mean).div_(self.std) # normalize
        return im

    @torch.no_grad()
    def random_hflip(self, im):
        '''
            input is nhwc
        '''
        imflip = im.flip(dims=(2,))
        probs = torch.rand(im.size(0))[:, None, None, None]
        probs = probs.to(im.device)
        res = torch.where(probs > 0.5, im, imflip)
        return res


def to_torch_func(x):
    return torch.from_numpy(np.array(x))

def get_dataset(traindir, aug_plus=True, n_views=2):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            #  transforms.RandomHorizontalFlip(),
            #  transforms.ToTensor(),
            #  normalize
            to_torch_func
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            #  transforms.RandomHorizontalFlip(),
            #  transforms.ToTensor(),
            #  normalize
            to_torch_func
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        #  TwoCropsTransform(transforms.Compose(augmentation)))
        NCropsTransform(transforms.Compose(augmentation), n_crops=n_views)
    )
    return train_dataset
