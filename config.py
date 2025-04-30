import argparse


def get_args(namespace=None):
    parser = argparse.ArgumentParser("Cross-Domain Few-Shot Learning")
    # --- global
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--write_to_local", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--git", action="store_true", help="local dev git repo?")

    # --- data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["RealLife", "DOLOS"],
    )
    parser.add_argument(
        "--num_frames",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--frame_size",
        default=224,
        type=int,
    )

    # --- model
    parser.add_argument(
        "--model",
        type=str,
        choices=["VideoMAE", "TransferOT", "W2V2_Model", "FusionModel", "ViT_model", "PECL"],
    )
    parser.add_argument(
        "--backbone_type",
        type=str,
        choices=["FusionModel", "W2V2_Model", "VideoMAE"],
        help="Only works when `args.model==TransferOT`",
    )
    parser.add_argument(
        "--pretrained_model_visual",
        default="pretrained_models/VideoMAE/",
        type=str,
    )
    parser.add_argument(
        "--pretrained_model_audio",
        default="pretrained_models/",
        type=str,
    )
    parser.add_argument(
        "--num_encoders",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
    )
    parser.add_argument(
        "--num_adapter",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--adapter_hidden_dim",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--adapter_type",
        default="efficient_conv",
        type=str,
        choices=["efficient_conv"],
    )
    parser.add_argument(
        "--hidden_dim",
        default=768,
        type=int,
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--feature_dim",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--hidden_dim_map",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--num_map_layer",
        default=2,
        type=int,
    )

    # --- solver
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        type=str,
        choices=["Adam", "SGD", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument(  # Weight decay has no effect when using Adam
        "--weight_decay",  # https://arxiv.org/abs/1711.05101
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--lr_factor",
        default=2,
        type=float,
    )
    parser.add_argument(
        "--weight_decay_bias",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--scheduler",
        default="StepLR",
        type=str,
        choices=["StepLR", "ReduceLROnPlateau"],
    )
    parser.add_argument(
        "--step_size",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--momentum",
        default=0.95,
        type=float,
    )
    parser.add_argument(
        "--scheduler_factor",
        default=0.1,
        type=float,
    )

    # --- train
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_final_ckpt", action="store_true")
    parser.add_argument("--save_best_ckpt", action="store_true")

    # --- transfer learning
    parser.add_argument(
        "--source_feature",
        type=str,
        help="This file should at least contain `features` and `labels` from source domain.",
    )
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--ot_max_iter", default=200, type=int)
    parser.add_argument("--ot_reduction", default=None)
    parser.add_argument("--xi", default=0.0, type=float)
    parser.add_argument("--delta", default=0.01, type=float)
    parser.add_argument("--thresh", default=1e-5, type=float)

    return parser.parse_args(namespace=namespace)
