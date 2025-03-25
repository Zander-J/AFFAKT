from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def make_optimizer(args, model, lr_factor=1.0):
    # return Adam(model.parameters(), lr=args.lr)
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        # if "bias" in key:
        #     lr = lr * args.lr_factor
        #     weight_decay = args.weight_decay_bias
        params += [
            {"params": [value], "lr": lr * lr_factor, "weight_decay": weight_decay}
        ]
    if args.optimizer == "SGD":
        optimizer = SGD(params, lr=args.lr, momentum=args.momentum)
    else:
        optimizer = eval(args.optimizer)(params, lr=args.lr)
    return optimizer


def make_lr_scheduler(args, optimizer):
    if args.scheduler == "StepLR":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_factor)

    elif args.scheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=args.scheduler_factor)

    else:
        raise KeyError("Invalid Scheduler Type")
