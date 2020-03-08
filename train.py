import torch


def compute_grad_norm(discriminator, data):
    data = data.requires_grad_()
    score = discriminator(data)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=data,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    grad_norm = (grad.norm(2, dim=1) ** 2).mean()
    return grad_norm


def train_one_round(
        loss_obj, discriminator, generator, d_optim, g_optim,
        data_loader_iter, noise_generator, fake_batch_size,
        num_step_d=1, num_step_g=1, simultaneous=True,
        real_grad_penalty=None, fake_grad_penalty=None,
        s=None, sparse_estimation=True,
        device=None,
):

    # simultaneous means that G and D are trained on the same fake batch
    #   and their parameters are updated simultaneously

    if (num_step_g > 1 or num_step_d > 1) and simultaneous:
        raise ValueError(num_step_g, num_step_d, simultaneous)

    if device is None:
        device = next(discriminator.parameters()).device

    p = generator.p

    noise_generator.eval()

    discriminator.eval()
    generator.train()
    lst_g_loss = []
    for ii in range(num_step_g):
        fake_data = generator(noise_generator((fake_batch_size, p)))
        g_loss = loss_obj.g_loss_fn(discriminator(fake_data))
        lst_g_loss.append(g_loss.item())
        g_optim.zero_grad()
        g_loss.backward()
        # _dim = generator.eta.size(0)
        # generator.eta.grad *= (_dim ** 0.5 / 10.)
        g_optim.step()

        if s is not None and sparse_estimation is True:
            with torch.no_grad():
                _, index = torch.topk(torch.abs(generator.eta), p - s, largest=False)
                generator.eta[index] = 0.

    discriminator.train()
    lst_d_loss = []
    generator.eval()
    for ii in range(num_step_d):
        # real_data, _ = next(data_loader_iter)
        real_data = next(data_loader_iter)[0]
        real_data = real_data.to(device)

        if simultaneous:
            fake_data = fake_data.detach()
        else:
            fake_data = generator(
                noise_generator((fake_batch_size, p))).detach()

        d_loss = loss_obj.d_loss_fn(discriminator(real_data),
                                    discriminator(fake_data))
        lst_d_loss.append(d_loss.item())

        if real_grad_penalty is not None:
            d_loss = d_loss + real_grad_penalty * compute_grad_norm(
                discriminator, real_data)

        if fake_grad_penalty is not None:
            d_loss = d_loss + fake_grad_penalty * compute_grad_norm(
                discriminator, fake_data)

        d_optim.zero_grad()
        d_loss.backward()


        # _dim = discriminator.layers[0].weight.size(1)
        # discriminator.layers[0].weight.grad *= _dim / 100.
        # discriminator.layers[0].bias.grad *= _dim / 100.

        d_optim.step()

    return lst_d_loss, lst_g_loss
