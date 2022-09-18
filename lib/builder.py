import torch
import torch.nn as nn

from .cutmix import CutMixer



class ModelWrapper(nn.Module):
    """
    Build a DenseCL model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/2011.09157
    """
    def __init__(self, base_model, dim=128, K=65536, m=0.999, T=0.07, mlp=False, cutmix=False, dense=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(ModelWrapper, self).__init__()

        self.cutmixer = CutMixer(T=T) if cutmix else None
        self.dense = dense

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_model(num_classes=dim, dense=dense, mlp=mlp)
        self.encoder_k = base_model(num_classes=dim, dense=dense, mlp=mlp)
        ## predictor
        #  mocov3-r50: fc-bn-relu-fc(bias=False)
        #  self.predictor = nn.Sequential(
        #          nn.Linear(dim, dim * 16, bias=False),
        #          nn.BatchNorm1d(dim * 16),
        #          nn.ReLU(inplace=True),
        #          nn.Linear(dim * 16, dim, bias=False)
        #          )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.crit_cls = nn.CrossEntropyLoss()

        # denseCL
        if dense:
            self.register_buffer("queue_dense", torch.randn(dim, K))
            self.queue_dense = nn.functional.normalize(self.queue_dense, dim=0)
            self.crit_dense = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, dense_keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        #  self.queue.mul_(0.999)
        self.queue[:, ptr:ptr + batch_size] = keys.T # replace the keys at ptr (dequeue and enqueue)

        # DenseCL
        if self.dense:
            dense_keys = nn.functional.normalize(dense_keys.mean(dim=2), dim=1)
            dense_keys = concat_all_gather(dense_keys)
            #  self.queue_dense.mul_(0.999)
            self.queue_dense[:, ptr:ptr + batch_size] = dense_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, k, feat_k, dense_k, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = k.shape[0]
        k_gather = concat_all_gather(k)
        batch_size_all = k_gather.shape[0]

        # restored index for this gpu
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        feat_k_gather, dense_k_gather = None, None
        if self.dense:
            feat_k_gather = concat_all_gather(feat_k)
            dense_k_gather = concat_all_gather(dense_k)
            feat_k_gather = feat_k_gather[idx_this]
            dense_k_gather = dense_k_gather[idx_this]


        return k_gather[idx_this], feat_k_gather, dense_k_gather

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, dense_q, feat_q = self.encoder_q(im_q)  # queries: NxC
        #  q = self.predictor(q)
        n, c, h, w = feat_q.size()
        dim_dense = dense_q.size(1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # inference key in shuffled order
            k, dense_k, feat_k = self.encoder_k(im_k)  # keys: NxC

            # undo shuffle
            k, feat_k, dense_k = self._batch_unshuffle_ddp(
                    k, feat_k, dense_k, idx_unshuffle)

            k = nn.functional.normalize(k, dim=1)
            if self.dense:
                dense_k, feat_k = dense_k.flatten(2), feat_k.flatten(2)
                dense_k_norm = nn.functional.normalize(dense_k, dim=1)
                ## match
                feat_q_norm = nn.functional.normalize(feat_q.flatten(2), dim=1)
                feat_k_norm = nn.functional.normalize(feat_k.flatten(2), dim=1)
                cosine = torch.einsum('nca,ncb->nab', feat_q_norm, feat_k_norm)
                #  cosine = feat_q_norm.permute(0, 2, 1) @ feat_k_norm

                pos_idx = cosine.argmax(dim=-1)
                dense_k_norm = dense_k_norm.gather(2, pos_idx.unsqueeze(1
                    ).expand(-1, dim_dense, -1))

        # compute logits
        q = nn.functional.normalize(q, dim=1)
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        #  l_pos = (q * k).sum(dim=1).unsqueeze(1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        #  l_neg = q @ self.queue.clone().detach()

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_cls = self.crit_cls(logits, labels)

        extra = {'qk': [q, k], 'logits': logits , 'labels': labels}

        ## densecl logits
        if self.dense:
            dense_q, feat_q = dense_q.flatten(2), feat_q.flatten(2)
            dense_q = nn.functional.normalize(dense_q, dim=1)

            d_pos = torch.einsum('ncm,ncm->nm', dense_q, dense_k_norm).unsqueeze(1)
            d_neg = torch.einsum('ncm,ck->nkm', dense_q, self.queue_dense.clone().detach())
            #  d_pos = (dense_q * dense_k_norm).sum(dim=1).unsqueeze(1)
            #  d_neg = self.queue_dense.clone().permute(1, 0).unsqueeze(0).detach() @ dense_q

            logits_dense = torch.cat([d_pos, d_neg], dim=1)
            logits_dense = logits_dense / self.T
            labels_dense = torch.zeros((n, h*w), dtype=torch.long).cuda()

            loss_dense = self.crit_dense(logits_dense, labels_dense)

            extra.update({'dense_qk': [dense_q, dense_k_norm],
                'loss_dense': loss_dense})

        ## regionCL
        if self.cutmixer:
            loss_cutmix = self.cutmixer.forward_mix(
                    self.encoder_q, im_q, q, k, self.queue.detach())
            extra.update({'loss_cutmix': loss_cutmix})

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, dense_k)

        return loss_cls, extra



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
