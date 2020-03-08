import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, p, initializer=None):
        super().__init__()
        self.p = p
        if initializer is None:
            self.eta = nn.Parameter(torch.randn(self.p) / self.p)
        else:
            self.eta = nn.Parameter(initializer)

    def forward(self, x):
        x = x + self.eta
        return x

    def get_numpy_eta(self):
        return self.eta.detach().cpu().numpy()


class Discriminator(nn.Module):

    def __init__(self, input_dim, lst_num_hidden, lst_activation, kappa=None,
                 l1_constrain_type=None):
        super().__init__()

        assert len(lst_activation) == len(lst_num_hidden)

        self.lst_activation = lst_activation
        self.kappa = kappa
        self.layers = nn.ModuleList()

        lst_num_nodes = [input_dim, ] + lst_num_hidden + [1, ]

        for i in range(len(lst_num_nodes) - 2):
            # fc = torch.nn.utils.spectral_norm(
            #     nn.Linear(lst_num_nodes[i], lst_num_nodes[i + 1]))
            # fc = L2RowConstrainedLinear(lst_num_nodes[i], lst_num_nodes[i + 1])
            fc = nn.Linear(lst_num_nodes[i], lst_num_nodes[i + 1])
            self.layers.append(fc)
        assert len(lst_activation) == len(self.layers)

        if self.kappa is None:
            self.last_linear_layer = nn.Linear(
                lst_num_nodes[-2], lst_num_nodes[-1], bias=False)
        else:
            assert (l1_constrain_type is not None)
            self.last_linear_layer = L1ConstrainedLinear(
                lst_num_nodes[-2], lst_num_nodes[-1], bias=False,
                l1_constrain_type=l1_constrain_type)

        # assert len(lst_num_hidden) == 1
        # self.bn = nn.BatchNorm1d(lst_num_hidden[0])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.lst_activation[i](x)

        # x = self.bn(x)

        if self.kappa is not None:
            x = self.last_linear_layer(x, self.kappa)
        else:
            x = self.last_linear_layer(x)

        return x
        # return torch.tanh(x)



class L1ConstrainedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 l1_constrain_type="proj"):
        super().__init__(in_features, out_features, bias)
        self.l1_constrain_type = l1_constrain_type

    def forward(self, input, kappa):
        # XXX: currently constraining the norm of the entire weight, maybe
        #   should just constrain each row
        if self.l1_constrain_type == "reparam":
            ww = _l1_scale(self.weight, kappa)
        elif self.l1_constrain_type == "scale":
            self.weight.data = _l1_scale(self.weight.data, kappa)
            ww = self.weight
        elif self.l1_constrain_type == "proj":
            self.weight.data = _l1_proj(self.weight.data, kappa)
            ww = self.weight
        else:
            raise ValueError(self.l1_constrain_type)

        return F.linear(input, ww, self.bias)


def _l1_scale(x, kappa):
    l1norm = x.abs().sum()
    # TODO: could be simplified with max
    if l1norm > kappa:
        return kappa * (x / l1norm)
    else:
        return x


def _l1_proj(x, kappa):
    from advertorch.utils import batch_l1_proj
    return batch_l1_proj(x[None, :], kappa)[0]


class L2RowConstrainedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 l2_constrain_type="reparam"):
        super().__init__(in_features, out_features, bias)
        self.l2_constrain_type = l2_constrain_type

    def forward(self, input, rownorm=1.):
        if self.l2_constrain_type == "reparam":
            ww = _batch_l2_scale(self.weight, rownorm)
        elif self.l2_constrain_type in ["scale", "proj"]:
            self.weight.data = _batch_l2_scale(self.weight.data, rownorm)
            ww = self.weight
        else:
            raise ValueError(self.l2_constrain_type)

        return F.linear(input, ww, self.bias)


def _batch_l2_scale(x, rownorm):
    from advertorch.utils import clamp_by_pnorm
    return clamp_by_pnorm(x, 2., rownorm)
