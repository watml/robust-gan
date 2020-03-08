import torch
import torch.nn as nn

from utils import coord_median, set_seed


class MMD(nn.Module):
    """Implementation of MMD GAN"""

    def __init__(self, sigma, device):
        super().__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, X, Y):
        """
        X: a tensor of size (n, p)
        Y: a tensor of size (m, p)
        """
        n = X.size(0)
        m = Y.size(0)

        return self.gaussian_kernel(X, X, denominator=self.sigma ** 2, diagonal=True) / n / (n - 1) \
            + self.gaussian_kernel(Y, Y, denominator=self.sigma ** 2, diagonal=True) / m / (m - 1) \
            - 2 * self.gaussian_kernel(X, Y, denominator=self.sigma ** 2, diagonal=False) / n / m

    @staticmethod
    def gaussian_kernel(X, Y, denominator, diagonal):
        """
        Helper function to compute the gaussian_kernel to each entry of X
        and sum them up
        X: a tensor of size (n, p)
        Y: a tensor of size (m, p)
        denominator:
        diagonal: whether subtract diagonal entries or not
        """
        ret = (- torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)).abs() ** 2, dim=-1)
               / 2 / denominator).exp().sum()

        if diagonal:
            ret -= (- torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)).abs() ** 2, dim=-1).diag()
                    / 2 / denominator).exp().sum()

        return ret


def test_mmd(args, device):
    mmd = MMD(lam=args.sigma,
              device=device)

    n = args.train_size

    X = torch.randn((n, args.p))
    Y = torch.randn((n, args.p)) + 1

    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        loss = mmd(X, Y)

    print('MMD loss = {:.4d}'.format(loss.item()))


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

    parser.add_argument("--sigma", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--debug", type=str2bool, default=0)

    parser.add_argument("--test", type=str2bool, default=0)

    parser.add_argument("--save_info_loc", type=str, default=None)


    args = parser.parse_args()

    print(args)

    set_seed(args.seed)

    if args.test:
        test_mmd(args, device)
        exit()

    theta = torch.zeros(args.p).to(device)

    data, theta = generate_contaminated_data(args.eps,
                                             args.train_size,
                                             theta=theta,
                                             type_cont=args.contamination,
                                             coord_median_as_origin=False)
    data = data.to(device)
    theta = theta.to(device)

    data_loader = torch.utils.data.DataLoader(TensorDataset(data),
                                              batch_size=args.real_batch_size,
                                              shuffle=True,
                                              num_workers=0)

    noise_generator = NoiseGenerator().to(device)

    '''
    Do not use coordinate-wise median as initialization.
    The global minimum of MMD GAN has mean square error very close to the coordinate-wise median,
    thus we prefer the training starting from somewhere else in order to see the progress of training.
    '''
    generator = Generator(p=args.p,
                          # initializer=torch.ones(args.p),
                          initializer=1.5 * coord_median(data),
                          ).to(device)

    mmd = MMD(sigma=args.sigma,
              device=device)

    g_optim = torch.optim.SGD(generator.parameters(),
                              lr=args.g_sgd_lr,
                              momentum=args.g_sgd_momentum)

    print('initial dist {:.4f}'.format(
        torch.norm(generator.eta - theta).item()))

    lst_eta = [generator.get_numpy_eta()]

    for i in range(args.num_epoch):
        total_loss = 0
        for batch_index, real_data in enumerate(data_loader):
            real_data = real_data[0].to(device)
            fake_data = generator(
                noise_generator((args.fake_batch_size, args.p)))

            loss = mmd(real_data, fake_data)

            g_optim.zero_grad()
            loss.backward()
            if args.g_sgd_normalize:
                with torch.no_grad():
                    generator.eta.grad /= torch.norm(generator.eta.grad)
            g_optim.step()

            total_loss += loss.item()

        lst_eta.append(generator.get_numpy_eta())

        total_loss /= (batch_index + 1)

        print('epoch {:3d}, dist {:.4f}, avg mmd {:.6f}'.format(i + 1,
                                                                torch.norm(generator.eta - theta).item(),
                                                                total_loss))

        if args.debug:
            print(generator.get_numpy_eta())

    if args.save_info_loc is not None:
        torch.save((theta.cpu().numpy(), lst_eta), args.save_info_loc)
        print("saved")
