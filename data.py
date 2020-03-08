import torch
import torch.nn as nn


def generate_dirty_data(type_cont, theta, num):
    def _extract_center():
        loc = float(type_cont.split("_")[1])
        theta_cont = loc * torch.ones_like(theta)
        return theta_cont


    def _generate_gauss_cov():
        from torch.distributions import Bernoulli
        from torch.distributions import Uniform

        ones = torch.ones((len(theta), len(theta)), device=theta.device)
        prec = Bernoulli(0.1 * ones).sample() * \
            Uniform(0.4 * ones, 0.8 * ones).sample()

        for ii in range(len(theta)):
            for jj in range(ii + 1, len(theta)):
                prec[ii, jj] = prec[jj, ii]

        prec = prec + (torch.symeig(prec)[0].min().abs() + 0.05) * torch.eye(
            len(theta), device=theta.device)

        return torch.inverse(prec)


    if type_cont.startswith("gauss"):
        dirty_data = _extract_center() + torch.randn(
            (num, len(theta)), device=theta.device)
    elif type_cont.startswith("cauchy"):
        from torch.distributions import Cauchy
        sampler = Cauchy(_extract_center(), torch.ones_like(theta))
        dirty_data = sampler.sample_n(num)
    elif type_cont.startswith("covgauss"):
        from torch.distributions import MultivariateNormal
        sampler = MultivariateNormal(
            _extract_center(),
            covariance_matrix=_generate_gauss_cov())
        dirty_data = sampler.sample_n(num)
    elif type_cont.startswith("dirac"):
        dirty_data = theta.new_ones((num, 1)).mm(_extract_center().view(1, -1))
    else:
        raise

    return dirty_data


def generate_contaminated_data(
        eps, num_data, theta=None,
        type_cont="gauss_5",
        coord_median_as_origin=True):

    if theta is None:
        dim = 100
        theta = torch.zeros(dim)

    randidx = torch.rand(num_data) < eps
    dirty_data = generate_dirty_data(type_cont, theta, randidx.sum().item())

    data = theta + torch.randn((num_data, len(theta)), device=theta.device)
    data[randidx] = dirty_data

    if coord_median_as_origin:
        from utils import coord_median
        coordmedian = coord_median(data)
        data = data - coordmedian
        theta = theta - coordmedian
    return data, theta


class NoiseGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = "cpu"

    def forward(self, size):
        return torch.randn(size=size, device=self.device)

    def to(self, device):
        self.device = device
        return self



class NoEndingDataLoaderIter(object):

    def __init__(self, loader):
        self.loader = loader
        self.iter = loader.__iter__()
        self.batch_size = loader.batch_size
        self.epoch = 0

    def __next__(self):

        try:
            return next(self.iter)
        except StopIteration:
            self.epoch += 1
            self.iter = self.loader.__iter__()
            return next(self.iter)
