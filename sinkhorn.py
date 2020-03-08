import torch
import torch.nn as nn

from utils import coord_median, set_seed


class SinkhornIteration(nn.Module):
    """
    Implementation of Sinkhorn itertion,
    allowing autodifferentiation to train generative models.
    """

    def __init__(self, lam, max_iter, device, const, thres):
        """
        lam: entropic regularization constant
        const: a small constant added to the cost matrix,
               to avoid non-differentiability of euclidean norm
        """
        super(SinkhornIteration, self).__init__()
        self.lam = lam
        self.max_iter = max_iter
        self.device = device
        self.const = const
        self.thres = thres

    def forward(self, X, Y):
        # Log domain sinkhorn iteration
        C = self.cost_matrix(X, Y)

        n_x = X.shape[0]
        n_y = Y.shape[0]

        # both marginals are fixed with uniform weights
        a = torch.ones(n_x, 1, dtype=torch.float,
                       device=self.device, requires_grad=False) / n_x
        b = torch.ones(n_y, 1, dtype=torch.float,
                       device=self.device, requires_grad=False) / n_y

        f = torch.zeros_like(a)
        g = torch.zeros_like(b)

        for i in range(self.max_iter):
            f_old = f

            K = self.log_boltzmann_kernel(C, f, g)
            f = self.lam * (torch.log(a)
                            - torch.logsumexp(K, dim=1, keepdim=True)) + f

            K = self.log_boltzmann_kernel(C, f, g)
            g = self.lam * (torch.log(b)
                            - torch.logsumexp(K.T, dim=1, keepdim=True)) + g

            if torch.norm(f - f_old).item() < self.thres:
                break

        if torch.norm(f - f_old).item() > self.thres:
            print('WARNING: sinkhorn iteration does not converge in {:d} iterations'.format(self.max_iter))

        pi = torch.exp(self.log_boltzmann_kernel(C, f, g))
        cost = torch.sum(pi * C)

        return cost, pi

    def log_boltzmann_kernel(self, C, f, g):
        kernel = -C + f + g.T
        kernel = kernel / self.lam
        return kernel

    def cost_matrix(self, X, Y):
        """
        Return the matrix of $||x_i-y_j||$.
        X: n * d tensor, each row is a data point
        Y: m * d tensor, each row is a data point
        """
        n, d1 = X.shape
        m, d2 = Y.shape

        assert d1 == d2

        X_col = X.unsqueeze(1)
        Y_lin = Y.unsqueeze(0)
        C = torch.sqrt(torch.sum((torch.abs(X_col - Y_lin)) ** 2, dim=2)
                       + self.const)

        assert C.shape == (n, m)

        return C


def lower_bound(theta, theta_cont, eta, epsilon):
    return torch.norm(eta
                      - (1 - epsilon) * theta
                      - epsilon * theta_cont
                      ).item()


def upper_bound(theta, theta_cont, eta, epsilon):
    return (1 - epsilon) * torch.norm(theta - eta).item() \
        + epsilon * torch.norm(theta_cont - eta).item()


def test_sinkhorn_iteration(args, device):
    sinkhorn = SinkhornIteration(lam=args.lam,
                                 max_iter=args.sinkhorn_max_iter,
                                 device=device,
                                 const=args.const,
                                 thres=args.thres)

    n = args.train_size

    # X = torch.tensor([[i, 0] for i in range(n)], dtype=torch.float)
    # Y = torch.tensor([[i, 1] for i in range(n)], dtype=torch.float)

    X = torch.randn((n, args.p))
    Y = torch.randn((n, args.p)) + 1

    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        dist, pi = sinkhorn(X, Y)

    print('dist = ', dist.item())
    print('Pi = ')
    print(pi)

    with torch.no_grad():
        print('cost matrix = ')
        print(sinkhorn.cost_matrix(X, Y))


if __name__ == '__main__':
    device = "cuda"

    from model import Generator
    from data import NoiseGenerator, generate_contaminated_data
    from torch.utils.data import TensorDataset
    from str2bool import str2bool

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--contamination", type=str, default="gauss_5")

    parser.add_argument("--real_batch_size", type=int, default=100)
    parser.add_argument("--fake_batch_size", type=int, default=100)

    parser.add_argument("--g_sgd_lr", type=float, default=0.001)
    parser.add_argument("--g_sgd_momentum", type=float, default=0.9)
    parser.add_argument("--g_sgd_normalize", type=str2bool, default=0)

    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sinkhorn_max_iter", type=int, default=50)
    parser.add_argument("--const", type=float, default=1e-6)
    parser.add_argument("--thres", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--debug", type=str2bool, default=0)

    parser.add_argument("--test", type=str2bool, default=0)

    parser.add_argument("--save_info_loc", type=str, default=None)

    args = parser.parse_args()

    print(args)

    if args.p == 1:
        assert args.const > 0

    set_seed(args.seed)

    if args.test:
        test_sinkhorn_iteration(args, device)
        exit()

    assert args.real_batch_size <= args.train_size

    theta = torch.zeros(args.p).to(device)

    data, theta = generate_contaminated_data(
        args.eps, args.train_size,
        theta=theta,
        type_cont=args.contamination,
        coord_median_as_origin=False)
    data = data.to(device)
    theta = theta.to(device)

    data_loader = torch.utils.data.DataLoader(
        TensorDataset(data),
        batch_size=args.real_batch_size, shuffle=True, num_workers=0)

    noise_generator = NoiseGenerator().to(device)

    '''
    We recommend not using coordinate-wise median as initialization.
    The global minimum of Wasserstein GAN has mean square error very close to the coordinate-wise median,
    thus we prefer the training starting from somewhere else in order to see the progress of training.
    '''
    generator = Generator(
        p=args.p,
        initializer=1.3 * coord_median(data_loader.dataset.tensors[0]),
        # 0.5 * torch.ones(args.p),
    ).to(device)

    sinkhorn = SinkhornIteration(lam=args.lam,
                                 max_iter=args.sinkhorn_max_iter,
                                 device=device,
                                 const=args.const,
                                 thres=args.thres)
    g_optim = torch.optim.SGD(generator.parameters(), lr=args.g_sgd_lr,
                              momentum=args.g_sgd_momentum)

    print('initial dist {:.4f}'.format(
        torch.norm(generator.eta - theta).item()))

    lst_eta = [generator.get_numpy_eta()]

    for i in range(args.num_epoch):
        total_cost = 0
        for batch_index, real_data in enumerate(data_loader):
            real_data = real_data[0].to(device)
            fake_data = generator(
                noise_generator((args.fake_batch_size, args.p)))

            cost, _ = sinkhorn(real_data, fake_data)

            g_optim.zero_grad()
            cost.backward()
            # print(generator.eta.grad)
            if args.g_sgd_normalize:
                with torch.no_grad():
                    generator.eta.grad /= torch.norm(generator.eta.grad)
            g_optim.step()

            total_cost += cost.item()

        total_cost /= (batch_index + 1)

        lst_eta.append(generator.get_numpy_eta())

        print('epoch {:3d},'.format(i + 1),
              'dist {:.4f},'.format(torch.norm(generator.eta - theta).item()),
              'avg Wass dist {:.4f},'.format(total_cost),
              'last Wass dist {:.4f},'.format(cost.item()))

        if args.debug:
            print(generator.get_numpy_eta())

    if args.save_info_loc is not None:
        torch.save((theta.cpu().numpy(), lst_eta), args.save_info_loc)
        print("saved")
