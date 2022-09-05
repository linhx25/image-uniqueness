import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src.model
import src.dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed=42, cuda_deterministic=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def pprint(*args):
    # print with current time
    time = "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] -"
    if torch.distributed.get_rank() == 0: ## DDP
        print(time, *args, flush=True)


def _freeze_modules(epoch, model, args):
    if (
        args.freeze_first_n_epochs == 0
        or epoch not in [0, args.freeze_first_n_epochs]
        or args.freeze_modules == ""
    ):
        return

    freeze = True
    if args.freeze_modules[0] == "~":
        freeze = False
        modules = args.freeze_modules[1:].split(",")
    else:
        modules = args.freeze_modules.split(",")

    if epoch == 0:
        pprint("..freeze modules:")
        grad = False
    else:
        pprint("..unfreeze modules:")
        grad = True

    for name, module in model.module.encoder_q.named_children():  ## DDP: model.module
        if (freeze and name in modules) or (not freeze and name not in modules):
            print(name, end=", ")
            for param in module.parameters():
                param.requires_grad_(grad)
    for name, module in model.module.encoder_k.named_children():  ## DDP: model.module
        if (freeze and name in modules) or (not freeze and name not in modules):
            print(name, end=", ")
            for param in module.parameters():
                param.requires_grad_(grad)
    print()


def load_state_dict_unsafe(model, state_dict):
    """
    Load state dict to provided model while ignore exceptions.
    """

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model)
    load = None  # break load->load reference cycle

    return {
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "error_msgs": error_msgs,
    }


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def init_memory_bank(model, data_loader, args):
    """Initialize memory bank"""
    if args.use_fullset:
        model.eval()

        for batch in tqdm(data_loader, desc="Initialize", total=len(data_loader), 
            disable=(args.local_rank % torch.cuda.device_count() != 0)):

            im_k = batch["image"][1].to(args.device)
            
            ## DDP: model.module
            k = model.module.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            model.module._dequeue_and_enqueue(k) # update memory bank
        model.module.queue_ptr[0] = 0 # don't change the pointer


# util
def setup_queue_size(data_loader, args, max_size=65536):
    """Set up size of queue, should be called before model setup"""
    ## DDP
    world_size = torch.distributed.get_world_size()
    batch_size = args.batch_size * world_size
    
    if args.use_fullset: # Prepare for fullset comparision
        size = min(max_size, len(data_loader.dataset))
        size = size - size % batch_size
        args.K = size
    else:
        args.K = args.K - args.K % batch_size
    pprint(f"Setup queue size: K={args.K}")


def get_image_ids(args, io=True):
    """
    Retrieve image ids in specific scenes.
    io: (True) use indoor/outdoor scenes
        (False) use place 365 scenes
    """
    if args.scene_dir == "": # dataset based on scene classification 
        return None
    else:
        try:
            args.scene = [int(s) for s in args.scene]
        except:
            io = False
        scene = pd.read_pickle(args.scene_dir)
        # indoor:0, outdoor:1, not_recognized:2
        if io:
            image_ids = scene[scene["in/out"].isin(args.scene)].index
        # bedroom, kitchen, bathroom, living_room
        else:
            image_ids = scene[scene["scene"].isin(args.scene)].index
        return image_ids


global_step = -1
def train_epoch(epoch, model, optimizer, scheduler, data_loader, writer, args):

    global global_step
    data_loader.sampler.set_epoch(epoch) ## DDP
    loss_fn = nn.CrossEntropyLoss().to(args.device)

    _freeze_modules(epoch, model, args)
    model.train()

    cnt = 0
    len_loop = min(args.max_iter_epoch, len(data_loader))

    for batch in tqdm(data_loader, total=len_loop, 
        disable=(args.local_rank % torch.cuda.device_count() != 0)):
        if cnt > args.max_iter_epoch:
            break
        cnt += 1
        global_step += 1

        images = batch["image"]
        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        output, target = model(im_q=images[0], im_k=images[1])
        loss = loss_fn(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()

        if writer:
            writer.add_scalar("Train/RunningLoss", loss.item(), global_step)
            writer.add_scalar("Train/Metric/%s" % "acc@1", acc1[0], global_step)
            writer.add_scalar("Train/Metric/%s" % "acc@5", acc5[0], global_step)

    scheduler.step()

    # save log
    if writer:
        for name, param in model.named_parameters():  # DDP: model.module or model
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(name + ".grad", param.grad, epoch)


def inference(model, data_loader, args, prefix="Test"):

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none").to(args.device)

    preds = []  # unique score = contrastive loss

    for batch in tqdm(
        data_loader, desc=prefix, total=len(data_loader), 
        disable=(args.local_rank % torch.cuda.device_count() != 0)):

        images, ids = batch["image"], batch["label"]
        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        with torch.no_grad():
            output, target = model(im_q=images[0], im_k=images[1])
            loss = loss_fn(output, target)

        loss = src.model.concat_all_gather(loss).cpu().numpy()
        if isinstance(ids, torch.Tensor):
            ids = ids.to(args.device)
            ids = src.model.concat_all_gather(ids).cpu().numpy()
        else:
            ids = src.model.concat_all_gather_object(ids)
            if isinstance(ids, pd.Series):
                ids = pd.Index(ids, names=["image_id"])
            else: # multi-index
                ids = pd.MultiIndex.from_frame(ids, names=["property_id", "image_id"])
        preds.append(pd.DataFrame({"loss": loss}, index=ids))

    preds = pd.concat(preds, axis=0)

    return preds


def main(args):

    torch.cuda.set_device(args.local_rank) ## DDP
    torch.distributed.init_process_group(backend="nccl") ## DDP
    set_seed(args.seed + args.local_rank, False)  ## DDP: args.seed + args.local_rank

    outdir = args.outdir + "/" + datetime.now().strftime("%m-%d_%H:%M:%S")
    if not os.path.exists(outdir) and args.local_rank == 0:
        os.makedirs(outdir)

    pprint("Configs: ", vars(args))

    # Transform
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([src.dataset.GaussianBlur([.1, 2.])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    transform_train = src.dataset.TwoCropsTransform(transform_train)

    # dataset
    train_ds = src.dataset.__dict__[args.dataset](
        args.data_dir, 
        transform=transform_train, 
        image_ids=get_image_ids(args, True),
    )

    # dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds) ## DDP
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        shuffle=(sampler is None),
        num_workers=args.n_workers,
    )

    # set up queue size to ensure divisble by batch size 
    setup_queue_size(train_loader, args)

    # model setting
    model = src.model.__dict__[args.model](
        torchvision.models.__dict__[args.arch], 
        K=args.K, 
        pretrained=args.pretrained, 
        projector_size=args.projector_size,
        ddp=True,
    )
    if args.init_state:
        pprint("load model init state")
        res = load_state_dict_unsafe(
            model, torch.load(args.init_state, map_location="cpu")
        )
        pprint(res)
    model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=(args.freeze_modules != ""),
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    ) ## DDP

    # initialize the model memory bank before training
    init_memory_bank(model, train_loader, args)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    writer = SummaryWriter(log_dir=outdir) if args.local_rank == 0 else None

    # training
    for epoch in range(args.n_epochs):

        pprint("Epoch:", epoch)

        pprint("training...")
        train_epoch(epoch, model, optimizer, scheduler, train_loader, writer, args=args)
        
        if (epoch % 4 == 0) and (args.local_rank == 0):
            torch.save(model.module.state_dict(), outdir + "/model.pt")  ## DDP: model.module

    # inference
    preds = inference(model, train_loader, args, "Inference-train")

    if args.local_rank == 0:
        preds.to_pickle(outdir + f"/train_loss.pkl")
        args.device = "cuda:0"
        info = dict(
            config=vars(args),
        )

        with open(outdir + "/info.json", "w") as f:
            json.dump(info, f, indent=4)
        pprint("finished.")


def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # model
    parser.add_argument("--out_feat", type=int, default=6)
    parser.add_argument("--init_state", default="")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--model", default="MoCo")
    parser.add_argument("--projector_size", type=int, default=[128], nargs="+")
    parser.add_argument("--K", type=int, default=65536, 
                        help="size of memory queue, should be divisible by batch_size")
    parser.add_argument("--use_fullset", action="store_true", 
                        help="whether to use all data for comparision")

    # training
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--freeze_modules", default="")
    parser.add_argument("--freeze_first_n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=-1)  # -1: no early stop
    parser.add_argument("--loss", default="CE")
    parser.add_argument("--metric", default="acc")
    parser.add_argument("--max_iter_epoch", type=int, default=200)
    parser.add_argument("--milestones", type=int, default=[8, 16, 24], nargs="+")

    # data
    parser.add_argument("--pin_memory", action="store_false")
    parser.add_argument("--dataset", type=str, default="AirbnbDataset")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="batch size per GPU; should be factor of model queue size K")
    parser.add_argument("--n_workers", type=int, default=0)

    # other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--scene_dir", default="", 
                        help="train based on scene classification")
    parser.add_argument("--scene", default=[0], nargs="+", 
                        help="indoor:0, outdoor:1, not_recognized:2, or 365 scenes")
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_ckpts", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    parser.add_argument("--comments", default="", 
                        help="add comments without indent and dash`-`")

    args = parser.parse_args()
    args.device = torch.device("cuda", args.local_rank)

    return args


if __name__ == "__main__":

    args = parse_args()
    main(args)