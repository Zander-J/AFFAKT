import ot
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist


class Sinkhorn_low_level(nn.Module):
    def __init__(self, eps, max_iter, reduction="none", thresh=1e-5):
        super(Sinkhorn_low_level, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.thresh = thresh

    def forward(self, x, y, mu):
        device = x.device
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = mu.to(device)

        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .to(device)
        )
        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        actual_nits = 0

        for _ in range(self.max_iter):
            u1 = u
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1

            if err.item() < self.thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))

        cost = torch.sum(pi * C, dim=(-2))
        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def _cost_matrix(self, x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = 1 - self.d_cosine(x_col, y_lin)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class Sinkhorn_high_level(nn.Module):
    def __init__(self, eps, max_iter, reduction="none", thresh=1e-5):
        super(Sinkhorn_high_level, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.thresh = thresh

    def forward(self, x, y, given_cost):
        device = x.device
        C = given_cost
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .to(device)
            .squeeze()
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .to(device)
            .squeeze()
        )
        if y_points == 1:
            nu = nu.unsqueeze(0)
        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        actual_nits = 0

        for _ in range(self.max_iter):
            u1 = u
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1

            U, V = u, v
            pi = torch.exp(self.M(C, U, V))

            if err.item() < self.thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))

        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def _cost_matrix(self, x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = 1 - self.d_cosine(x_col, y_lin)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class OptimalTransportReg(nn.Module):
    def __init__(self) -> None:
        super(OptimalTransportReg, self).__init__()

    def forward(self, embed1, embed2, pred, label):
        _, s1, _ = torch.svd(embed1)
        _, s2, _ = torch.svd(embed2)
        sdist = self.dist(s1.reshape(-1, 1), s2.reshape(-1, 1))

        C0 = cdist(
            embed1.cpu().detach().numpy(),
            embed2.cpu().detach().numpy(),
            metric="sqeuclidean",
        )
        C1 = self.loss_hinge(
            label.reshape(-1, 1).cpu().detach().numpy(), pred.cpu().detach().numpy()
        )
        C = 100 * C0 + C1
        OUTM = ot.unif(embed1.shape[0])
        OUTT = ot.unif(embed2.shape[0])
        gamma = ot.emd(OUTM, OUTT, C)
        gamma = np.float32(gamma)
        gamma = torch.from_numpy(gamma).to(embed1.device)
        gdist = self.dist(embed1, embed2)
        t = torch.eye(2, dtype=pred.dtype, device=pred.device)[label, :]
        ldist = self.dist(pred, t)
        return 0.001 * torch.sum(gamma * (gdist + ldist + sdist))

    def loss_hinge(self, Y, F):
        res = np.zeros((Y.shape[0], F.shape[0]))
        for i in range(Y.shape[1]):
            res += (
                np.maximum(
                    0,
                    1
                    - Y[:, i].reshape((Y.shape[0], 1))
                    * F[:, i].reshape((1, F.shape[0])),
                )
                ** 2
            )
        return res

    def dist(self, x, y=None):
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist


class OptimalTransport(nn.Module):
    r"""
    Cost matrix is learnable.
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(
        self,
        num_bclass,
        num_nclass,
        eps: float = 0.01,
        max_iter: int = 1000,
        reduction="none",
    ):
        super(OptimalTransport, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.C = nn.Parameter(
            torch.ones((num_bclass, num_nclass)) / 2, requires_grad=True
        )

    def forward(self, mu, nu):
        device = mu.device
        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        actual_nits = 0
        thresh = 1e-1

        for i in range(self.max_iter):
            u1 = u
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1

            U, V = u, v
            pi = torch.exp(self.M(U, V))

            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(U, V))

        res = torch.einsum("sbn,sb->sn", (pi, mu))

        return res

    def M(self, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-self.C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


# class OptimalTransport:
#     def __init__(self, entreg: float = 0.5, max_iter: int = 1000) -> None:
#         self.entreg = entreg
#         self.max_iter = max_iter

#     def ot(self, base_features, novel_features, base_dataset=None):
#         M = ot.dist(base_features, novel_features, metric="cosine")
#         if base_dataset == "K400":
#             base_dist = K400_DATASET_DIST
#         else:
#             base_dist = ot.unif(len(base_features))
#         novel_dist = ot.unif(len(novel_features))

#         P = ot.sinkhorn(
#             base_dist,
#             novel_dist,
#             M,
#             reg=self.entreg,
#             numItermax=self.max_iter,
#             method="sinkhorn_log",
#         )
#         return P


if __name__ == "__main__":
    import numpy as np

    ot_tool = OptimalTransport()
    base_features = np.random.rand(400, 768)
    novel_features = np.random.rand(2, 768)
    print(ot_tool.ot(base_features, novel_features, base_dataset="K400"))
