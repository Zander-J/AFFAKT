import torch
import torch.nn as nn
import torchaudio

from collections import namedtuple

try:
    from . import MODELS
    from .adapter import w2v2_adapter_conv
except ImportError:
    from models import MODELS
    from models.adapter import w2v2_adapter_conv


@MODELS.register
class W2V2_Model(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        ####### kwargs #######
        num_encoders: int = 4,
        adapter: bool = True,
        adapter_type: str = "efficient_conv",
        hidden_dim: int = 768,
        adapter_hidden_dim: int = 32,
        pretrained_model_audio: str = None,
        **kwargs,
    ):  # adapter_conv_params as a tuple (kernel_size, stride)
        super(W2V2_Model, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        model = torchaudio.pipelines.WAV2VEC2_BASE.get_model(
            dl_kwargs=dict(model_dir=pretrained_model_audio)
        )
        for p in model.parameters():
            p.requires_grad = False

        # pretrained CNN feature extracor
        self.FEATURE_EXTRACTOR = model.feature_extractor

        # pretrained feature projection + pos encoding
        self.FEATURE_PROJECTOR = nn.Sequential(
            model.encoder.feature_projection,
            model.encoder.transformer.pos_conv_embed,
            model.encoder.transformer.layer_norm,
            model.encoder.transformer.dropout,
        )

        # build w2V2 encoder with desired number of encoder layers
        layer_list = []

        for i in range(self.num_encoders):
            if self.adapter:
                if self.adapter_type == "mlp":
                    raise NotImplementedError(
                        "MLP adapter is not implemented in VideoMAE, please use efficient_conv instead."
                    )
                elif self.adapter_type == "efficient_conv":
                    layer_list.append(
                        w2v2_adapter_conv(
                            transformer_encoder=model.encoder.transformer.layers[i],
                            hidden_dim=hidden_dim,
                            adapter_hidden_dim=adapter_hidden_dim,
                        )
                    )
            else:
                # fine_tune enoder in case we donot use adapters
                for p in model.encoder.transformer.layers[i].parameters():
                    p.requires_grad = True
                layer_list.append(model.encoder.transformer.layers[i])

        self.TRANSFORMER = nn.Sequential(*layer_list)

        # linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.num_classes),
            nn.Softmax(),
        )

        self.mapping = nn.Sequential(nn.Linear(self.hidden_dim, self.feature_dim))
        self.result = namedtuple("Result", ["output", "loss"])
        self.loss_fn = nn.CrossEntropyLoss(reduce="mean")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def type(self):
        return next(self.parameters()).dtype

    def forward(self, data_collection, train_stage=False):
        label = data_collection.label.to(self.device)
        output_tokens = self.extract_feature(data_collection)
        logits = self.classifier(output_tokens)
        loss = self.loss_fn(logits, label)

        return self.result(output=logits, loss=loss)

    def extract_feature(self, data_collection):
        features, _ = self.FEATURE_EXTRACTOR(
            data_collection.audio.squeeze(dim=1).to(self.device), None
        )
        projections = self.FEATURE_PROJECTOR(features)
        output_tokens = self.TRANSFORMER(projections)
        return self.mapping(output_tokens).mean(axis=1)
