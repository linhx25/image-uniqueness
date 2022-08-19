# Pytorch implementation of personalized MoCo
# Based on https://github.com/facebookresearch/moco/blob/main/moco/builder.py
import torch
import torch.nn as nn
import pandas as pd


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, 
            base_encoder, 
            dim=128, 
            K=65536, 
            m=0.999, 
            T=0.07, 
            pretrained=False, 
            projector_size=[128],
            ddp=False,
        ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        pretrained: whether load pretrained weights
        projector_size: hidden size of projection head
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.ddp = ddp

        # create the encoders
        self.encoder_q = base_encoder(pretrained=pretrained)
        self.encoder_k = base_encoder(pretrained=pretrained)
        
        # replace projection head (ResNet)
        projection_head = self.encoder_q.fc
        in_feat = projection_head.in_features
        projector_size[-1] = dim
        
        self.encoder_q.fc = self._build_mlp(in_feat, projector_size)
        self.encoder_k.fc = self._build_mlp(in_feat, projector_size)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        # gather keys before updating queue
        keys = concat_all_gather(keys) if self.ddp else keys

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x) if self.ddp else x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        if self.ddp:
            # broadcast to all gpus
            torch.distributed.broadcast(idx_shuffle, src=0)

            # shuffled index for this gpu
            gpu_idx = torch.distributed.get_rank()
            idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

            return x_gather[idx_this], idx_unshuffle
        else:
            return x_gather, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x) if self.ddp else x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        if self.ddp:
            # restored index for this gpu
            gpu_idx = torch.distributed.get_rank()
            idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

            return x_gather[idx_this]
        else:
            return x_gather

    def _build_mlp(self, in_feat, projector_size):
        layers = []
        for i, size in enumerate(projector_size):
            if i == 0:
                layers.append(nn.Linear(in_feat, size, bias=False))
            else:
                layers.append(nn.Linear(projector_size[i-1] , size))
            if i != len(projector_size) - 1:
                layers.append(nn.BatchNorm1d(size))    
                layers.append(nn.ReLU(inplace=True))
            else: # SimCLR
                layers.append(nn.BatchNorm1d(size, affine=False))
        return nn.Sequential(*layers)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCo_VGG(MoCo):
    def __init__(self, 
            base_encoder, 
            dim=128, 
            K=65536, 
            m=0.999, 
            T=0.07, 
            pretrained=False, 
            projector_size=[4096, 4096, 128],
            ddp=False,
        ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        pretrained: whether load pretrained weights
        train_module: name of projection head in base_encoder
        projection_layer: number of layers of projection head
        projector_size: hidden size of projection head
        """
        super(MoCo, self).__init__() # do not inherit MoCo.__init__
        self.K = K
        self.m = m
        self.T = T
        self.ddp = ddp

        # create the encoders
        self.encoder_q = base_encoder(pretrained=pretrained)
        self.encoder_k = base_encoder(pretrained=pretrained)
        
        # replace projection head (VGG)
        projection_head = self.encoder_q.classifier
        if isinstance(projection_head, nn.Sequential):
            in_feat = list(projection_head.children())[0].in_features
        else:
            in_feat = projection_head.in_features
        projector_size[-1] = dim
        
        self.encoder_q.classifier = self._build_mlp(in_feat, projector_size)
        self.encoder_k.classifier = self._build_mlp(in_feat, projector_size)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def concat_all_gather_object(obj):
    """
    Performs all_gather operation on the provided object: List[Tuple].
    """
    obj_list = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(obj_list, obj)

    if pd.DataFrame(obj_list[0]).shape[1] == 1: # one-dim
        output = [pd.Series(obj) for obj in obj_list] 
        output = pd.concat(output)
    else:
        output = [pd.DataFrame(obj).T for obj in obj_list] 
        output = pd.concat(output, axis=0)
    return output


if __name__ == "__main__":
    import torchvision.models as models
    import argparse

    args = argparse.ArgumentParser(allow_abbrev=False)
    args.add_argument("--local_rank", type=int, default=0)
    args.add_argument("--ddp", action="store_true")
    args = args.parse_args()

    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)

        model = MoCo(models.__dict__["alexnet"], ddp=args.ddp)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

        X1 = torch.randn(8, 3, 224, 224)  # batchsize * channel * height * width
        X2 = torch.randn(8, 3, 224, 224)
        output, target = model(im_q=X1, im_k=X2)

        if args.local_rank == 0:
            print(model, "\n")
            print(output.shape)
            print(target.shape)

    else:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        model = MoCo(models.__dict__["alexnet"])

        X1 = torch.randn(8, 3, 224, 224)  # batchsize * channel * height * width
        X2 = torch.randn(8, 3, 224, 224)

        output, target = model(im_q=X1, im_k=X2)

        print(model, "\n")
        print(output.shape)
        print(target.shape)
