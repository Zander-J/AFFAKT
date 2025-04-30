import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import h5py
import time
import json
import time
import warnings
from git import Repo

warnings.filterwarnings("ignore")

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel


from models import MODELS
from models.build import get_model_args
from data import DATASETS
from data.build import get_dataset_args, get_dataloader
from solver.build import make_optimizer, make_lr_scheduler
from config import get_args
from utils.miscellaneous import (
    set_seed,
    make_output_dir,
    get_rank,
    get_world_size,
    schedule_xi,
    setup_for_distributed,
    reduce_tensor,
    save_model,
)
from utils.logger import get_logger
from utils.metrics import get_metrics, AverageMeter, Metrics
from utils.miscellaneous import reduce_metrics, update_meter, accumulate_meters


def train_one_epoch(model, train_loader, optim, amp, tbar):
    model.train()
    scaler = GradScaler()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for data_collection in train_loader:
        label = data_collection.label.to(model.device)
        optim.zero_grad()
        if amp:
            with autocast():
                res = model(data_collection, train_stage=True)
                output, loss = res.output, res.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            res = model(data_collection)
            output, loss = res.output, res.loss
            loss.backward()
            optim.step()

        # Microsoft Swin-Transformer
        # https://github.com/microsoft/Swin-Transformer/blob/main/main_simmim_pt.py#L158
        torch.cuda.synchronize()

        acc = get_metrics(output.detach(), label, training=True).accuracy
        acc = reduce_tensor(acc)
        loss = reduce_tensor(loss)
        acc_meter.update(acc.item(), label.size(0))
        loss_meter.update(loss.item(), label.size(0))
        if tbar is not None:
            tbar.set_postfix(loss="%.4f" % loss.item(), acc="%.4f" % acc.item())
            tbar.update()

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_old(model, test_loader, amp):
    """evaluation

    Based on Microsoft Swin-Transformer
    https://github.com/microsoft/Swin-Transformer/blob/main/main_simmim_ft.py#L222
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    f1_meter = AverageMeter()
    auc_meter = AverageMeter()
    for data_collection in test_loader:
        label = data_collection.label.to(model.device)
        if amp:
            with autocast():
                res = model(data_collection)
        else:
            res = model(data_collection)
        output, loss = res.output, res.loss
        res = get_metrics(output, label, training=True)
        acc = reduce_tensor(res.accuracy)
        f1 = reduce_tensor(res.binary_f1_score)
        auc = reduce_tensor(res.micro_auc)
        loss = reduce_tensor(loss)
        acc_meter.update(acc.item(), label.size(0))
        f1_meter.update(f1.item(), label.size(0))
        auc_meter.update(auc.item(), label.size())
        loss_meter.update(loss.item(), label.size(0))
    return loss_meter.avg, (f1_meter.avg, acc_meter.avg, auc_meter.avg)


@torch.no_grad()
def evaluate(model, test_loader, amp, tbar):
    """evaluation

    Based on Microsoft Swin-Transformer
    https://github.com/microsoft/Swin-Transformer/blob/main/main_simmim_ft.py#L222
    """
    model.eval()
    outputs = []
    labels = []
    loss_meter = AverageMeter()
    tp_meter = AverageMeter()
    meters = [AverageMeter() for _ in range(len(Metrics._fields))]

    for data_collection in test_loader:
        label = data_collection.label
        if amp:
            with autocast():
                res = model(data_collection)
        else:
            res = model(data_collection)
        labels.append(label.cpu().data)
        outputs.append(res.output.cpu().data)
        loss_meter.update(res.loss.cpu().item(), label.size(0))
        if tbar is not None:
            tbar.update()
    torch.cuda.synchronize()
    labels = torch.cat(labels, dim=0)
    outputs = torch.cat(outputs, dim=0)
    res = get_metrics(outputs, labels)
    try:
        res = reduce_metrics(res)
    except RuntimeError:
        pass
    update_meter(meters, res, labels.size(0))
    res = accumulate_meters(meters)
    return loss_meter.avg, (res.binary_f1_score, res.accuracy, res.micro_auc)


def train(args):
    # torch DDP setup
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.num_gpus = get_world_size()
        args.batch_size = args.batch_size * args.num_gpus
        args.lr = args.lr * args.num_gpus
        setup_for_distributed(local_rank == 0)
        torch.distributed.barrier()
    else:
        args.distributed = False
        local_rank = 0
        args.num_gpus = 1
        device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    set_seed(args.seed)
    if local_rank == 0:
        args.output_dir = make_output_dir(args, get_rank())
        log = get_logger(args, get_rank())
        log.info(
            f'Experiment time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}',
        )
        log.info(str(args))
        if args.git:
            for i in Repo(".").iter_commits():
                sha = i.hexsha
                break
            log.info("Git HexSHA of this code is: " + sha)
        pass
    assert args.model in MODELS
    dataset = DATASETS[args.dataset](**get_dataset_args(args))

    if local_rank == 0:
        index_file = None
        if args.write_to_local:
            index_file = h5py.File(os.path.join(args.output_dir, "train_test_index.hdf5"), "w")
            index_file.create_dataset("Name", shape=(1,), data=args.dataset, dtype="S10")
        log.info("Start training...")
        if not (args.save_final_ckpt or args.save_best_ckpt):
            log.warn("Checkpoints would NOT be saved.")
        summary = dict()

    # Stratified K-Fold validation.
    splits = StratifiedKFold(n_splits=args.k_fold, random_state=args.seed, shuffle=True)
    for fold_idx, (train_set, test_set) in enumerate(
        splits.split(dataset.clip_files, dataset.labels)
    ):
        if local_rank == 0:
            summary[f"fold_{fold_idx}"] = dict()
            if args.write_to_local:
                index_file.create_dataset(f"fold_{fold_idx}/train", data=train_set)
                index_file.create_dataset(f"fold_{fold_idx}/test", data=test_set)

        model = MODELS[args.model](**get_model_args(args))
        model.to(device)
        if args.distributed:
            if local_rank == 0:
                log.info(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPUs.")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        optim = make_optimizer(args, model, args.batch_size)
        scheduler = make_lr_scheduler(args, optim)

        best_acc = 0
        best_epoch = 0

        train_loader, test_loader = get_dataloader(dataset, train_set, test_set, args)
        PRINT_STEP = 1
        for epoch in range(args.num_epochs):
            # train one epoch
            if args.model == "TransferOT":
                xi = schedule_xi(args.xi, epoch, args.num_epochs, mode="cosine")
                try:
                    model.module.set_xi(xi)  # only for transfer_ot
                except AttributeError:
                    model.set_xi(xi)
            # get dataloader for each epoch
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            tbar = None
            if local_rank == 0:
                tbar = tqdm(
                    total=len(train_loader),
                    desc=f"Fold {fold_idx} Epoch {epoch}",
                )
            loss_train, acc_train = train_one_epoch(model, train_loader, optim, args.amp, tbar)
            if (epoch + 1) % PRINT_STEP == 0:
                # evaluate after training this epoch
                tbar = None
                if local_rank == 0:
                    tbar = tqdm(
                        total=len(test_loader),
                        desc=f"Fold {fold_idx} Epoch {epoch}",
                    )
                loss_test, (f1_test, acc_test, auc_test) = evaluate(
                    model, test_loader, args.amp, tbar
                )
            if args.distributed:
                torch.distributed.barrier()
            if local_rank == 0 and (epoch + 1) % PRINT_STEP == 0:
                log.info(
                    f"@@@ Fold {fold_idx} Epoch {epoch + 1} Train: loss {np.mean(loss_train)}, acc {acc_train} "
                    f"Test: loss {np.mean(loss_test)}, acc {acc_test}",
                )
                # Update best model checkpoint and acc.
                if acc_test > best_acc:
                    best_acc = acc_test
                    best_epoch = epoch + 1
                    prefix = ""
                    if args.write_to_local and args.save_best_ckpt:
                        save_model(
                            model,
                            os.path.join(
                                args.output_dir,
                                "checkpoints",
                                f"fold_{fold_idx}_best.pth",
                            ),
                            args.model,
                        )
                        prefix = "Update checkpoint. "
                    log.info(prefix + f"Best acc: {acc_test}")
                    summary[f"fold_{fold_idx}"]["best_acc"] = best_acc
                    summary[f"fold_{fold_idx}"]["best_f1"] = f1_test
                    summary[f"fold_{fold_idx}"]["best_auc"] = auc_test
                    summary[f"fold_{fold_idx}"]["best_epoch"] = best_epoch
            if args.scheduler == "ReduceLROnPlateau":
                scheduler.step(acc_test, epoch=epoch)
            else:
                scheduler.step()
        # finish training all epoch.
        if args.model == "TransferOT":
            try:
                assert model.module.xi == args.xi
            except AttributeError:
                assert model.xi == args.xi
        if args.distributed:
            torch.distributed.barrier()
        if local_rank == 0:
            prefix = ""
            if args.write_to_local and args.save_final_ckpt:
                save_model(
                    model,
                    os.path.join(args.output_dir, "checkpoints", f"fold_{fold_idx}_final.pth"),
                    args.model,
                )
                prefix = "Final model saved. "
            # Evaluate final model.
            if best_acc == 0:
                loss_test, (f1_test, acc_test, auc_test) = evaluate(model, test_loader, args.amp)
            summary[f"fold_{fold_idx}"]["final_acc"] = acc_test
            summary[f"fold_{fold_idx}"]["best_f1"] = f1_test
            summary[f"fold_{fold_idx}"]["best_auc"] = auc_test
            prefix_best = f"Best acc {best_acc} at epoch {best_epoch}. "
            log.info(
                f"### Fold {fold_idx}: {prefix}{prefix_best}Final acc {acc_test}.",
            )
    if local_rank == 0:
        if args.write_to_local:
            index_file.close()
        try:
            all_acc = [summary[key]["best_acc"] for key in summary.keys() if "fold" in key]
            summary["acc_mean"] = np.mean(all_acc)
            summary["acc_std"] = np.std(all_acc)
            all_f1 = [summary[key]["best_f1"] for key in summary.keys() if "fold" in key]
            summary["f1_mean"] = np.mean(all_f1)
            summary["f1_std"] = np.std(all_f1)
            all_auc = [summary[key]["best_auc"] for key in summary.keys() if "fold" in key]
            summary["auc_mean"] = np.mean(all_auc)
            summary["auc_std"] = np.std(all_acc)
        except KeyError:
            summary["acc_mean"] = "None"
            summary["acc_std"] = "None"
            summary["f1_mean"] = "None"
            summary["f1_std"] = "None"
            summary["auc_mean"] = "None"
            summary["auc_std"] = "None"
        all_acc = [summary[key]["final_acc"] for key in summary.keys() if "fold" in key]
        summary["final_acc_mean"] = np.mean(all_acc)
        summary["final_acc_std"] = np.std(all_acc)

        log.info(f"Finsh training. Summary:\n{json.dumps(summary)}")
    if args.distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ in "__main__":
    train(get_args())
