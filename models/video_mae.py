import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Set, Tuple, Union

from transformers import VideoMAEForVideoClassification
from collections import namedtuple

try:
    from models.adapter import (
        video_mae_adapter_layer_conv,
        video_mae_adapter_layer_conv_multi,
    )
    from models import MODELS
    from models.optimal_transport import OptimalTransport, OptimalTransportReg
except ImportError:
    from .adapter import (
        video_mae_adapter_layer_conv,
        video_mae_adapter_layer_conv_multi,
    )
    from . import MODELS
    from .optimal_transport import OptimalTransport, OptimalTransportReg


class VideoMAEEncoderAdapter(nn.Module):
    def __init__(
        self,
        num_encoder,
        num_adapter,
        hidden_dim,
        encoder,
        adapter,
        adapter_type,
        adapter_hidden_dim,
    ) -> None:
        super(VideoMAEEncoderAdapter, self).__init__()
        self.encoder = encoder
        layer_list = []
        if adapter:
            if num_adapter == num_encoder:
                for i in range(num_encoder):
                    if adapter_type == "mlp":
                        raise NotImplementedError(
                            "MLP adapter is not implemented in VideoMAE, please use efficient_conv instead."
                        )
                    elif adapter_type == "efficient_conv":
                        layer_list.append(
                            video_mae_adapter_layer_conv(
                                transformer_encoder_layer=encoder.layer[i],
                                hidden_dim=hidden_dim,
                                adapter_hidden_dim=adapter_hidden_dim,
                            )
                        )
                    else:
                        raise NotImplementedError(
                            f"Such method has not beed implemented."
                        )
            elif num_encoder % num_adapter == 0:
                block = num_encoder // num_adapter
                for i in range(0, num_encoder, block):
                    if adapter_type == "mlp":
                        raise NotImplementedError(
                            "MLP adapter is not implemented in VideoMAE, please use efficient_conv instead."
                        )
                    elif adapter_type == "efficient_conv":
                        layer_list.append(
                            video_mae_adapter_layer_conv_multi(
                                transformer_encoder_layers=encoder.layer[
                                    i * block : (i + 1) * block
                                ],
                                hidden_dim=hidden_dim,
                                adapter_hidden_dim=adapter_hidden_dim,
                            )
                        )
                    else:
                        raise NotImplementedError(
                            f"Such method has not beed implemented."
                        )
            else:
                raise RuntimeError(
                    f"Number of encoders must can be divided by the number of adapters."
                )
        else:
            for i in range(num_encoder):
                for p in encoder.layers[i].parameters():
                    p.requires_grad = True
                layer_list.append(encoder.layers[i])
        self.layer = nn.Sequential(*layer_list)

    def forward(self, hidden_states):
        return self.layer(hidden_states)


@MODELS.register
class VideoMAE(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        ####### kwargs ########
        num_encoders: int = 12,
        num_frames: int = 16,
        hidden_dim: int = 768,
        pretrained_model_visual: str = None,
        # adapter
        adapter: bool = True,
        num_adapter: int = 12,
        adapter_type: str = "efficient_conv",
        adapter_hidden_dim: int = 32,
        **kwargs,
    ) -> None:
        super(VideoMAE, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.num_adapter = num_adapter
        self.feature_dim = feature_dim
        self.loss_cls = nn.CrossEntropyLoss(reduce="mean")

        self.videoMAEModel = VideoMAEForVideoClassification.from_pretrained(
            pretrained_model_visual
        )
        assert self.hidden_dim == self.videoMAEModel.videomae.config.hidden_size
        for _, p in self.videoMAEModel.named_parameters():
            p.requires_grad = False

        self.embedding = self.videoMAEModel.videomae.embeddings

        self.encoder = VideoMAEEncoderAdapter(
            self.num_encoders,
            self.num_adapter,
            self.hidden_dim,
            self.videoMAEModel.videomae.encoder,
            adapter,
            adapter_type,
            adapter_hidden_dim,
        )
        self.mapping = nn.Sequential(nn.Linear(self.hidden_dim, self.feature_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes),
            nn.Softmax(),
        )
        self.loss_cls = nn.CrossEntropyLoss(reduce="mean")
        self.result = namedtuple("Result", ["output", "loss"])

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def type(self):
        return next(self.parameters()).dtype

    def forward(self, data_collection):
        label = data_collection.label.to(self.device)
        feature_map = self.extract_feature(data_collection)

        output = self.classifier(feature_map)

        output = F.softmax(output, dim=1)
        loss = self.loss_cls(output, label)

        return self.result(output=output, loss=loss)

    def extract_feature(self, data_collection):
        feature_map = data_collection.visual.to(self.device)
        feature_map = self.embedding(feature_map, None)

        feature_map = self.encoder(hidden_states=feature_map)

        if self.videoMAEModel.fc_norm is not None:
            feature_map = self.videoMAEModel.fc_norm(feature_map.mean(dim=1))
        else:
            feature_map = feature_map[:, 0]

        return self.mapping(feature_map)
