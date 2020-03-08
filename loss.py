import torch
import torch.nn as nn


class Loss(object):

    def __init__(self):
        self.zeros_by_batch_size = {}
        self.ones_by_batch_size = {}

    def g_loss_fn(self, fake_score):
        raise NotImplementedError()

    def d_loss_fn(self, real_score, fake_score):
        raise NotImplementedError()

    def get_ones(self, ref_data):
        batch_size = len(ref_data)
        device = ref_data.device
        return self.ones_by_batch_size.setdefault(
            (batch_size, device.type, device.index),
            torch.ones((batch_size, 1), device=device)
        )

    def get_zeros(self, ref_data):
        batch_size = len(ref_data)
        device = ref_data.device
        return self.zeros_by_batch_size.setdefault(
            (batch_size, device.type, device.index),
            torch.zeros((batch_size, 1), device=device)
        )


class JSLoss(Loss):

    def __init__(self):
        super().__init__()
        self._loss_fn = nn.BCEWithLogitsLoss()

    def g_loss_fn(self, fake_score):
        return self._loss_fn(fake_score, self.get_ones(fake_score))

    def d_loss_fn(self, real_score, fake_score):
        return (self._loss_fn(real_score, self.get_ones(real_score)) +
                self._loss_fn(fake_score, self.get_zeros(fake_score)))


class TVLoss(Loss):

    def __init__(self):
        super().__init__()

        def _loss_fn(score):
            return torch.mean(torch.sigmoid(score))

        self._loss_fn = _loss_fn


    def g_loss_fn(self, fake_score):
        return -self._loss_fn(fake_score)

    def d_loss_fn(self, real_score, fake_score):
        return -self._loss_fn(real_score) + self._loss_fn(fake_score)


class KLLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -torch.mean(torch.exp(fake_score))

    def d_loss_fn(self, real_score, fake_score):
        return -torch.mean(real_score + 1) + torch.mean(torch.exp(fake_score))


class RKLLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -torch.mean(fake_score - 1)

    def d_loss_fn(self, real_score, fake_score):
        return (-torch.mean(-torch.exp(-real_score))
                + torch.mean(fake_score - 1))


class SHLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -torch.mean(torch.exp(fake_score) - 1)

    def d_loss_fn(self, real_score, fake_score):
        return (-torch.mean(1 - torch.exp(-real_score))
                + torch.mean(torch.exp(fake_score) - 1))


class WrongSHLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -torch.mean(fake_score * torch.exp(fake_score))

    def d_loss_fn(self, real_score, fake_score):
        return (-torch.mean(1 - torch.exp(-real_score))
                + torch.mean(fake_score * torch.exp(fake_score)))


class WSLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -torch.mean(fake_score)

    def d_loss_fn(self, real_score, fake_score):
        return -torch.mean(real_score) + torch.mean(fake_score)


class QuarterWSLoss(Loss):

    def __init__(self):
        super().__init__()

    def g_loss_fn(self, fake_score):
        return -0.25 * torch.mean(fake_score)

    def d_loss_fn(self, real_score, fake_score):
        # print(real_score.max().abs().item(), fake_score.max().abs().item())
        # print(real_score.abs().mean().item(), fake_score.abs().mean().item())
        return -0.25 * torch.mean(real_score) + 0.25 * torch.mean(fake_score)
