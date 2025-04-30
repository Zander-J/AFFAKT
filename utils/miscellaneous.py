import os
import torch
import numpy as np
import random
import torch.distributed as dist

from utils.metrics import Metrics


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_output_dir(args, local_rank):
    if local_rank > 0:
        return
    dir = args.output_dir
    if not args.write_to_local:
        return dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    _dirs = [i for i in os.listdir(dir) if i.startswith("exp")]
    if len(_dirs) == 0:
        path = os.path.join(dir, "exp")
    else:
        mm = -1
        for _dir in _dirs:
            d = _dir[3:]
            try:
                d = int(d)
            except ValueError:
                d = 0
            mm = max(mm, d)
        path = os.path.join(dir, "exp" + str(mm + 1))
    os.makedirs(path)
    os.makedirs(os.path.join(path, "checkpoints"))
    return path


def get_rank():
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_world_size():
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def schedule_xi(xi, epoch, num_epoch, mode="linear"):
    if mode == "linear":
        alpha = 1.0 * epoch / (num_epoch - 1)
        return xi * alpha
    elif mode == "cosine":
        alpha = -np.cos(1.0 * epoch / (num_epoch - 1) * np.pi) + 1
        return xi * alpha / 2
    pass


def save_model(model, path, model_type):
    try:
        params_to_save = dict(model=(model.module.state_dict()))
        if model_type == "TransferOT":
            params_to_save.update({"prototype": model.module.prototype})
    except AttributeError:
        params_to_save = dict(model=(model.state_dict()))
        if model_type == "TransferOT":
            params_to_save.update({"prototype": model.prototype})
    torch.save(params_to_save, path)
    pass


def load_model(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    try:
        model.prototype = ckpt["prototype"]
    except:
        pass
    return model


def reduce_metrics(res):
    kwargs = dict()
    for key in res._fields:
        value = getattr(res, key)
        try:
            value = reduce_tensor(value)
            torch.distributed.barrier()
        except RuntimeError:
            pass
        kwargs.update({key: value})
    return Metrics(**kwargs)


def update_meter(meters, res, n=1):
    for idx, key in enumerate(res._fields):
        value = getattr(res, key)
        meters[idx].update(value.item(), n)
    try:
        torch.distributed.barrier()
    except RuntimeError:
        pass


def accumulate_meters(meters):
    return Metrics(*[item.avg for item in meters])
