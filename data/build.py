import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from transformers import VideoMAEImageProcessor

try:
    from data.path_categ import DATASET_CONFIG
    from data.utils import af_collate_fn, reg_collate
    from data import DATASETS
except ImportError:
    from .path_categ import DATASET_CONFIG
    from .utils import af_collate_fn, reg_collate
    from . import DATASETS


def get_dataset_args(args):
    assert args.dataset in DATASETS
    params = DATASET_CONFIG[args.dataset]
    params.update(n_sample_frames=args.num_frames)
    params.update(frame_size=args.frame_size)

    if args.dataset == "RealLife":
        params.update(modalities=["visual", "audio"])
        pass
    elif args.dataset == "DOLOS":
        params.update(modalities=["visual", "audio"])
        pass
    else:
        raise KeyError
    if args.model in {"VideoMAE", "TransferOT", "FusionModel"}:
        params.update(
            transform=VideoMAEImageProcessor.from_pretrained(args.pretrained_model_visual)
        )
    else:
        params.update(
            transform=T.Compose(
                [
                    T.ToTensor(),
                    T.ConvertImageDtype(torch.float32),
                    T.Resize((args.frame_size, args.frame_size), antialias=True),
                    # normalize to imagenet mean and std values
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        )
    return params


def get_dataloader(dataset, train_idx, test_idx, args):
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_sampler, test_sampler = None, None
    if args.distributed:
        train_sampler = DistributedSampler(
            train_set,
            rank=args.local_rank,
            shuffle=True,
            num_replicas=args.num_gpus,
            seed=args.seed,
        )
        test_sampler = DistributedSampler(
            test_set,
            rank=args.local_rank,
            shuffle=False,
            num_replicas=args.num_gpus,
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        num_workers=0,
        collate_fn=af_collate_fn,
        drop_last=True,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=2 * args.num_gpus,
        shuffle=False,
        num_workers=0,
        collate_fn=af_collate_fn,
        drop_last=False,
        sampler=test_sampler,
    )
    return train_loader, test_loader
