import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def coord_median(x):
    return torch.median(x, dim=0)[0]


def plot_data(X, theta):
    """
    X: numpy array
    theta: numpy array
    """
    plt.scatter(X[:, 0], X[:, 1], c='yellow', s=60, edgecolor='k')
    plt.scatter(theta[0], theta[1], marker='*', c='red',
                s=240, edgecolor='k', label='True Mean')


def plot_db_discriminator(discriminator, x_min, x_max, y_min, y_max):
    device = next(discriminator.parameters()).device

    ep = 1
    xx, yy = np.meshgrid(np.arange(x_min - ep, x_max + ep, 0.01),
                         np.arange(y_min - ep, y_max + ep, 0.01))

    points = np.column_stack((xx.ravel(), yy.ravel()))
    with torch.no_grad():
        zz = discriminator(torch.tensor(
            points, dtype=torch.float, device=device)).detach().cpu().numpy()
    zz = zz.reshape(xx.shape)

    # plt.contourf(xx, yy, zz, cmap = plt.cm.binary, alpha = 0.5)
    # cp = plt.contourf(xx, yy, zz, alpha = 0.9)
    cp = plt.contourf(xx, yy, zz)
    plt.colorbar(cp)


def plot_generator(generator):
    theta_hat = generator.get_numpy_eta()
    plt.scatter(theta_hat[0], theta_hat[1], marker='^',
                c='green', s=180, edgecolor='k', label='Generator')


def plot_visualization(discriminator, generator, dataloader, theta, device):
    discriminator.eval()
    generator.eval()

    X_numpy = dataloader.dataset.tensors[0].detach().cpu().numpy()

    theta_numpy = theta.detach().cpu().numpy()
    theta_hat = generator.get_numpy_eta()

    plot_db_discriminator(
        discriminator, x_min=min(np.min(X_numpy[:, 0]), np.min(theta_hat)),
        x_max=max(np.max(X_numpy[:, 0]), np.max(theta_hat)),
        y_min=min(np.min(X_numpy[:, 1]), np.min(theta_hat)),
        y_max=max(np.max(X_numpy[:, 1]), np.max(theta_hat)))
    plot_data(X_numpy, theta_numpy)
    plot_generator(generator)

    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)


def initialize_d_optimizer(params, args):
    if args.d_optimizer == "sgd":
        d_optim = torch.optim.SGD(
            params, lr=args.d_sgd_lr, momentum=args.d_sgd_momentum,
            weight_decay=args.sgd_weight_decay)
    elif args.d_optimizer == "adam":
        d_optim = torch.optim.Adam(
            params, lr=args.d_adam_lr, betas=(args.d_adam_b1, args.d_adam_b2),
            weight_decay=args.adam_weight_decay)
    elif args.d_optimizer == "adagrad":
        d_optim = torch.optim.Adagrad(
            params,
            lr=args.d_adagrad_lr,
            lr_decay=args.d_adagrad_lr_decay,
            weight_decay=args.adagrad_weight_decay,
            initial_accumulator_value=args.d_adagrad_initial_accumulator_value,
        )
    else:
        raise

    return d_optim


def initialize_g_optimizer(params, args):
    if args.g_optimizer == "sgd":
        g_optim = torch.optim.SGD(
            params, lr=args.g_sgd_lr, momentum=args.g_sgd_momentum)
    elif args.g_optimizer == "adam":
        g_optim = torch.optim.Adam(
            params, lr=args.g_adam_lr, betas=(args.g_adam_b1, args.g_adam_b2))
    else:
        raise

    return g_optim


def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
