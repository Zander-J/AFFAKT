import torch
import torch.nn as nn

from collections import namedtuple

try:
    from . import MODELS
    from .video_mae import VideoMAE
    from .audio_model import W2V2_Model
except ImportError:
    from models import MODELS
    from models.video_mae import VideoMAE
    from .audio_model import W2V2_Model


@MODELS.register
class FusionModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        num_encoders: int = 12,
        num_frames: int = 16,
        hidden_dim: int = 768,
        pretrained_model_visual: str = None,
        pretrained_model_audio: str = None,
        adapter: bool = True,
        num_adapter: int = 12,
        adapter_type: str = "efficient_conv",
        adapter_hidden_dim: int = 32,
        **kwargs
    ) -> None:
        super(FusionModel, self).__init__()

        self.visual_model = VideoMAE(
            num_encoders=num_encoders,
            num_frames=num_frames,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            num_classes=num_classes,
            pretrained_model_visual=pretrained_model_visual,
            adapter=adapter,
            num_adapter=num_adapter,
            adapter_type=adapter_type,
            adapter_hidden_dim=adapter_hidden_dim,
            **kwargs
        )
        self.audio_model = W2V2_Model(
            num_encoders=num_encoders,
            adapter=adapter,
            adapter_type=adapter_type,
            hidden_dim=hidden_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            feature_dim=feature_dim,
            num_classes=num_classes,
            pretrained_model_audio=pretrained_model_audio,
            **kwargs
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
            nn.Softmax(),
        )
        self.result = namedtuple("Result", ["output", "loss"])
        self.loss_fn = nn.CrossEntropyLoss(reduce="mean")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def type(self):
        return next(self.parameters()).dtype

    def forward(self, data_collection):
        label = data_collection.label.to(self.device)
        feature = self.extractor_feature(data_collection)
        output = self.classifier(feature)
        loss = self.loss_fn(output, label)
        return self.result(output=output, loss=loss)

    def extract_feature(self, data_collection):
        visual_feature = self.visual_model.extract_feature(data_collection)
        audio_feature = self.audio_model.extract_feature(data_collection)
        fused_feature = 0.5 * visual_feature + 0.5 * audio_feature
        return fused_feature
