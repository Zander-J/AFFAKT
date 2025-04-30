try:
    from models import MODELS
except ImportError:
    from . import MODELS


def get_model_args(args):
    assert args.model in MODELS
    if args.model == "VideoMAE":
        params = dict(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_encoders=args.num_encoders,
            num_frames=args.num_frames,
            hidden_dim=args.hidden_dim,
            pretrained_model_visual=args.pretrained_model_visual,
            adapter=args.adapter,
            num_adapter=args.num_adapter,
            adapter_type=args.adapter_type,
            adapter_hidden_dim=args.adapter_hidden_dim,
        )
    elif args.model == "TransferOT":
        params = dict(
            source_feature=args.source_feature,
            epsilon=args.epsilon,
            max_iter=args.ot_max_iter,
            reduction=args.ot_reduction,
            xi=args.xi,
            delta=args.delta,
            thresh=args.thresh,
            hidden_dim_map=args.hidden_dim_map,
            nu=args.nu,
            num_map_layer=args.num_map_layer,
            backbone_type=args.backbone_type,
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_encoders=args.num_encoders,
            num_adapter=args.num_adapter,
            adapter=args.adapter,
            adapter_type=args.adapter_type,
            hidden_dim=args.hidden_dim,
            num_frames=args.num_frames,
            pretrained_model_visual=args.pretrained_model_visual,
            pretrained_model_audio=args.pretrained_model_audio,
            adapter_hidden_dim=args.adapter_hidden_dim,
            alpha=args.alpha,
        )
    elif args.model == "FusionModel":
        params = dict(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_encoders=args.num_encoders,
            num_frames=args.num_frames,
            hidden_dim=args.hidden_dim,
            pretrained_model_visual=args.pretrained_model_visual,
            pretrained_model_audio=args.pretrained_model_audio,
            adapter=args.adapter,
            num_adapter=args.num_adapter,
            adapter_type=args.adapter_type,
            adapter_hidden_dim=args.adapter_hidden_dim,
        )
    elif args.model == "W2V2_Model":
        params = dict(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_encoders=4,  # use default setting in ICCV 2023 paper
            adapter=args.adapter,
            adapter_type=args.adapter_type,
            hidden_dim=args.hidden_dim,
            adapter_hidden_dim=args.adapter_hidden_dim,
            pretrained_model_audio=args.pretrained_model_audio,
        )
    elif args.model == "ViT_model":
        params = dict(
            num_encoders=4,  # use default setting in ICCV 2023 paper
            adapter=args.adapter,
            adapter_type=args.adapter_type,
        )
    elif args.model == "PECL":
        params = dict()
    else:
        raise NotImplementedError(f"Model {args.model} has not been implemented.")
    return params
