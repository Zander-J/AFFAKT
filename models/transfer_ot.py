import h5py
import numpy as np
from collections import Counter, namedtuple
from geomloss import SamplesLoss

from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from . import MODELS
    from .video_mae import VideoMAE
    from .audio_model import W2V2_Model
    from .fusion_model import FusionModel
    from .optimal_transport import Sinkhorn_low_level, Sinkhorn_high_level
    from .map_model import Map
except ImportError:
    from models import MODELS
    from models.video_mae import VideoMAE
    from models.audio_model import W2V2_Model
    from models.fusion_model import FusionModel
    from models.optimal_transport import Sinkhorn_low_level, Sinkhorn_high_level
    from models.map_model import Map


@MODELS.register
class TransferOT(nn.Module):
    def __init__(
        self,
        ######## Source ########
        source_feature,
        ########## OT ##########
        epsilon: float = 0.01,
        max_iter: int = 200,
        reduction: str = None,  # None, "mean", "sum"
        xi: float = 0.2,
        delta: float = 0.5,
        thresh: float = 1e-5,
        num_map_layer: int = 2,
        hidden_dim_map: int = 128,
        nu: float = 0.1,
        alpha: float = 0.95,
        ####### backbone #######
        backbone_type: str = "FusionModel",
        feature_dim: int = 512,
        num_classes: int = 2,
        **kwargs
    ) -> None:
        super(TransferOT, self).__init__()
        self.xi = xi
        self.trans = (xi - 0.0) > 1e-8
        self.delta = delta
        self.ot_thresh = thresh
        self.num_classes = num_classes
        self.nu = nu
        self.alpha = alpha

        self.target_model = eval(backbone_type)(
            feature_dim=feature_dim,
            num_classes=num_classes,
            **kwargs,
        )

        if self.trans:
            self.load_source_feature(source_feature)
            self.sinkhorn_low_level = Sinkhorn_low_level(
                eps=epsilon,
                max_iter=max_iter,
                reduction=reduction,
                thresh=self.ot_thresh,
            )
            self.sinkhorn_high_level = Sinkhorn_high_level(
                eps=epsilon,
                max_iter=max_iter,
                reduction=reduction,
                thresh=self.ot_thresh,
            )

            self.knowledge_tran = Map(
                in_dim=feature_dim,
                out_dim=feature_dim,
                hidden_dim=hidden_dim_map,
                n_layer=num_map_layer,
            )
            self.target_mapping = Map(
                in_dim=feature_dim,
                out_dim=feature_dim,
                hidden_dim=hidden_dim_map,
                n_layer=num_map_layer,
            )
            self.source_mapping = Map(n_layer=0)
            self.loss_map = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
            self.prototype = torch.ones(
                (num_classes, len(self.source_classes)), requires_grad=False
            ) / len(self.source_classes)
        else:
            self.prototype = None
            pass

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
            nn.Softmax(),
        )
        self.loss_cls = nn.CrossEntropyLoss(reduction="mean")
        self.result = namedtuple("Result", ["output", "loss"])

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if self.trans:
            self.prototype = self.prototype.to(device)
            self.source_feature = self.source_feature.to(device)
            self.sample_probabi = self.sample_probabi.to(device)
            self.source_mean = self.source_mean.to(device)
        return super(TransferOT, self).to(*args, **kwargs)

    def forward(self, data_collection, train_stage=False):
        label = data_collection.label.to(self.device)
        # target model
        target_feature = self.target_model.extract_feature(data_collection)

        # knowledge transfer
        if self.trans:
            target_feature = self.target_mapping(target_feature)
            source_feature = self.source_mapping(self.source_feature.to(self.dtype))
            cost_inner, pi, _ = self.sinkhorn_low_level(
                source_feature, target_feature.unsqueeze(0), self.sample_probabi
            )
            source_mean = source_feature.mean(axis=1)
            _, pi, _ = self.sinkhorn_high_level(
                source_mean,
                target_feature,
                cost_inner,
            )
            # \sum_{b\in B} T_{bn} = 1 / N  boundary constraints
            pi = pi.permute(1, 0).to(self.dtype) * len(label)
            tp = 0
            if train_stage:
                self.update_prototype(pi=pi, label=label)
            else:
                tp = torch.clamp(torch.std(pi, dim=1, keepdim=True) - self.nu, 0, 1)
                pi = (1 - tp) * self.select_prototype(pi=pi) + tp * pi
            tran = (pi.unsqueeze(-1) * source_mean.unsqueeze(0)).sum(axis=1)
            tran = self.knowledge_tran(tran)

            # fusion
            output = self.classifier((1 - self.xi) * target_feature + self.xi * tran)
            loss = self.loss_cls(output, label) + self.delta * self.loss_map(
                target_feature, source_mean
            )
            res = self.result(output=output, loss=loss)
            return res
        else:  # No transfer leanring
            output = self.classifier(target_feature)
            loss = self.loss_cls(output, label)
            res = self.result(output=output, loss=loss)

        return res

    @torch.no_grad()
    def update_prototype(self, pi, label):
        for cls in range(self.num_classes):
            mask = label == cls
            if not torch.any(mask).item():
                continue
            self.prototype[cls] = self.alpha * self.prototype[cls] + (1 - self.alpha) * pi[
                mask, :
            ].mean(axis=0)

    @torch.no_grad()
    def select_prototype(self, pi):
        k = torch.cdist(pi, self.prototype).argmin(axis=1)
        return self.prototype[k, :]

    def load_source_feature(self, source_feature_path):
        with h5py.File(source_feature_path, "r") as f:
            data = f["features"][:]
            label = f["labels"][:]

        classifer_source = LogisticRegression(max_iter=1000).fit(X=data, y=label)

        source_label_count = Counter(label.tolist())
        max_item = max(source_label_count.values())

        source_classes = set(source_label_count.keys())
        self.sample_probabi = torch.zeros([len(source_classes), max_item])
        source_mean = []
        feature_all_reshape = np.zeros(shape=[len(source_classes), max_item, data.shape[1]])
        for source_class in source_classes:
            mask = label == source_class
            feature = data[mask, ...]
            source_mean.append(np.mean(feature, axis=0))

            if len(feature) == max_item:
                feature_all_reshape[source_class, ...] = feature
            else:
                pad = max_item - len(feature)
                tmp = np.concatenate([feature, feature[:pad]], axis=0)
                while pad >= len(feature):
                    pad -= len(feature)
                    tmp = np.concatenate([tmp, feature[:pad]], axis=0)
                feature_all_reshape[source_class, ...] = tmp
            predict_prob = classifer_source.predict_proba(feature_all_reshape[source_class, ...])
            self.sample_probabi[source_class, ...] = F.softmax(
                torch.from_numpy(predict_prob[:, source_class] / 0.3), dim=0
            )
        self.source_feature = torch.from_numpy(feature_all_reshape)
        self.source_classes = list(source_classes)
        self.source_labels = torch.from_numpy(label)
        self.source_mean = self.source_feature.mean(axis=1)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_xi(self, xi):
        self.xi = xi
