import torch.nn as nn


class Map(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, hidden_dim=128, n_layer=2) -> None:
        super(Map, self).__init__()
        assert n_layer >= 0
        if n_layer == 0:
            self.model = nn.Sequential(nn.Identity())
        elif n_layer == 1:
            self.model = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            model_list = [nn.Linear(in_dim, hidden_dim)]
            for i in range(1, n_layer):
                model_list += [
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, out_dim if i == n_layer - 1 else hidden_dim),
                ]
            self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)
