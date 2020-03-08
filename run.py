import torch
import torch.nn as nn


if __name__ == '__main__':
    device = "cuda"

    from statistics import mean
    from str2bool import str2bool
    import matplotlib.pyplot as plt

    import loss
    from model import Generator, Discriminator
    from data import NoiseGenerator, generate_contaminated_data
    from data import NoEndingDataLoaderIter
    from utils import coord_median, plot_visualization
    from torch.utils.data import TensorDataset
    from train import train_one_round
    from utils import set_seed, initialize_d_optimizer, initialize_g_optimizer

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=150)
    parser.add_argument("--num_iter", type=int, default=-1)
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--s", type=int, default=-1)
    parser.add_argument("--sparse_estimation", type=str2bool, default=True)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--coord_median_as_origin", type=str2bool, default=1)
    parser.add_argument("--contamination", type=str, default="gauss_5")

    parser.add_argument("--loss", type=str, default="JSLoss")
    parser.add_argument("--kappa", type=eval, default=None)
    parser.add_argument("--l1_constrain_type", type=str, default="scale")

    parser.add_argument("--real_batch_size", type=int, default=500)
    parser.add_argument("--fake_batch_size", type=int, default=500)
    parser.add_argument("--debug", type=str2bool, default=0)

    parser.add_argument("--simultaneous", type=str2bool, default=1)
    parser.add_argument("--num_step_d", type=int, default=1)
    parser.add_argument("--num_step_g", type=int, default=1)

    parser.add_argument("--d_optimizer", type=str, default="adam")
    parser.add_argument("--g_optimizer", type=str, default="sgd")

    parser.add_argument("--d_sgd_lr", type=float, default=0.02)
    parser.add_argument("--d_sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_weight_decay", type=float, default=0)

    parser.add_argument("--d_adam_lr", type=float, default=0.0002)
    parser.add_argument("--d_adam_b1", type=float, default=0.5)
    parser.add_argument("--d_adam_b2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0)

    parser.add_argument("--d_adagrad_lr", type=float, default=0.01)
    parser.add_argument("--d_adagrad_lr_decay", type=float, default=0)
    parser.add_argument("--d_adagrad_initial_accumulator_value",
                        type=float, default=0.0)
    parser.add_argument("--adagrad_weight_decay", type=float, default=0)

    parser.add_argument("--g_sgd_lr", type=float, default=0.02)
    parser.add_argument("--g_sgd_momentum", type=float, default=0.0)


    parser.add_argument("--g_adam_lr", type=float, default=0.0002)
    parser.add_argument("--g_adam_b1", type=float, default=0.5)
    parser.add_argument("--g_adam_b2", type=float, default=0.999)

    parser.add_argument("--real_grad_penalty", type=float, default=None)
    parser.add_argument("--fake_grad_penalty", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)


    args = parser.parse_args()

    print(args)

    if args.s == -1:
        args.s = None

    assert not (args.num_epoch == -1 and args.num_iter == -1)
    # assert args.real_batch_size <= args.train_size
    if args.debug and args.p != 2:
        raise ValueError(args.debug, args.p)

    assert isinstance(args.kappa, (int, float, dict, type(None)))
    if isinstance(args.kappa, dict):
        init_kappa = args.kappa[0]
    else:
        init_kappa = args.kappa

    set_seed(args.seed)

    if args.s is None:
        theta = torch.zeros(args.p).to(device)
    else:
        theta = torch.zeros(args.p).to(device)
        theta[0:args.s] = 1.

    data, theta = generate_contaminated_data(
        args.eps, args.train_size,
        theta=theta,
        type_cont=args.contamination,
        coord_median_as_origin=args.coord_median_as_origin)
    data = data.to(device)
    theta = theta.to(device)

    data_loader = torch.utils.data.DataLoader(
        TensorDataset(data),
        batch_size=args.real_batch_size, shuffle=True, num_workers=0)

    lst_activation = [nn.Sigmoid()]
    lst_num_hidden = [20]

    loss_obj = getattr(loss, args.loss)()
    noise_generator = NoiseGenerator().to(device)
    generator = Generator(
        p=args.p,
        initializer=coord_median(data_loader.dataset.tensors[0]),
        # initializer = torch.ones(args.p) * 2
    ).to(device)
    discriminator = Discriminator(
        input_dim=args.p,
        lst_num_hidden=lst_num_hidden,
        lst_activation=lst_activation,
        kappa=init_kappa,
        l1_constrain_type=args.l1_constrain_type,
    ).to(device)

    d_optim = initialize_d_optimizer(discriminator.parameters(), args)
    g_optim = initialize_g_optimizer(generator.parameters(), args)

    print("dist {:.4f}".format(torch.norm(generator.eta - theta).item()))

    data_loader_iter = NoEndingDataLoaderIter(data_loader)


    # g_scheduler = torch.optim.lr_scheduler.StepLR(
    #     g_optim, step_size=1, gamma=0.98)
    # d_scheduler = torch.optim.lr_scheduler.StepLR(
    #     d_optim, step_size=1, gamma=0.98)

    epoch = 0
    idx_iter = 0

    lst_eta = [generator.get_numpy_eta()]
    while True:
        idx_iter += 1

        if isinstance(args.kappa, dict):
            if epoch in args.kappa.keys():
                discriminator.kappa = args.kappa[epoch]
                print("Set kappa to {}".format(discriminator.kappa))
                del args.kappa[epoch]

        # XXX: note that training does not stop exactly at the end of the epoch

        lst_d_loss, lst_g_loss = train_one_round(
            loss_obj, discriminator, generator, d_optim, g_optim,
            data_loader_iter, noise_generator,
            fake_batch_size=args.fake_batch_size,
            device=None,
            real_grad_penalty=args.real_grad_penalty,
            fake_grad_penalty=args.fake_grad_penalty,
            num_step_d=args.num_step_d, num_step_g=args.num_step_g,
            simultaneous=args.simultaneous,
            s=args.s,
            sparse_estimation=args.sparse_estimation,
        )

        if data_loader_iter.epoch > epoch:
            lst_eta.append(generator.get_numpy_eta())
            # d_scheduler.step()
            # g_scheduler.step()
            print(
                "epoch {:6d},".format(epoch),
                "dist {:.4f},".format(
                    torch.norm(generator.eta - theta).item()),
                "d_loss {:.4f},".format(mean(lst_d_loss)),
                "g_loss {:.4f},".format(mean(lst_g_loss)),
                # *["norm {:.4f},".format(param.norm()) for param in discriminator.parameters()]
            )
            epoch = data_loader_iter.epoch

            if args.num_epoch != -1 and \
                    data_loader_iter.epoch >= args.num_epoch:
                break

            if args.debug:
                fig = plt.figure()
                fig.set_size_inches((10, 8))
                plot_visualization(
                    discriminator, generator, data_loader, theta,
                    device=None)
                title = 'Epoch ' + str(epoch)
                plt.title(title, fontsize=15)
                fig.savefig('./Figure/' + title + '.png')
                plt.close()

                print(generator.get_numpy_eta())

        if args.num_iter != -1 and idx_iter > args.num_iter:
            break


    torch.save((theta.cpu().numpy(), lst_eta), "results.pkl")
    print("saved")
