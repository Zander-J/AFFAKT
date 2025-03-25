""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
Copied from timm.utils.metrics.
"""

import torch

from collections import namedtuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) / batch_size for k in topk
    ]


def precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [
        correct[: min(k, maxk)]
        .view(-1)
        .float()
        .sum(0, keepdim=True)
        .mul_(100.0 / batch_size)
        for k in topk
    ]


Metrics = namedtuple(
    "Metrics",
    [
        "micro_precision",
        "macro_precision",
        "micro_recall",
        "macro_recall",
        "micro_f1_score",
        "macro_f1_score",
        "binary_f1_score",
        "accuracy",
        "micro_auc",
        "macro_auc",
    ],
)


def get_metrics(output, label, training=False):
    device = label.device
    output, label = output.cpu(), label.cpu()
    pred = output.argmax(dim=1)
    label_oh = torch.zeros((label.size(0), label.max() + 1), dtype=label.dtype).scatter(
        1, label.view(-1, 1), 1
    )
    if training:
        return Metrics(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.tensor(
                accuracy_score(label, pred), dtype=torch.float32, device=device
            ),
            None,
            None,
        )
    return Metrics(
        torch.tensor(
            precision_score(label, pred, average="micro"),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            precision_score(label, pred, average="macro"),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            recall_score(label, pred, average="micro"),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            recall_score(label, pred, average="macro"),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            f1_score(label, pred, average="micro"), dtype=torch.float32, device=device
        ),
        torch.tensor(
            f1_score(label, pred, average="macro"), dtype=torch.float32, device=device
        ),
        torch.tensor(
            f1_score(label, pred, average="binary"), dtype=torch.float32, device=device
        ),
        torch.tensor(accuracy_score(label, pred), dtype=torch.float32, device=device),
        torch.tensor(
            roc_auc_score(label_oh, output, average="micro", multi_class="ovr"),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            roc_auc_score(label_oh, output, average="macro", multi_class="ovr"),
            dtype=torch.float32,
            device=device,
        ),
    )
