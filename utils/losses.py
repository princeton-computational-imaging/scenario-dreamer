import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricLoss(nn.Module):
    def __init__(self, mean_dim=1, apply_mean=True):
        super().__init__()
        self.mean_dim = mean_dim
        self.apply_mean = apply_mean

    def forward(self, pred, targ, batch):
        loss = self._loss(pred, targ)

        num_samples = int(batch.max().detach()) + 1
        # Calculate the mean loss for each sample
        batch_size = batch.size(0)
        loss_per_sample = torch.zeros(num_samples, device=loss.device, dtype=loss.dtype)
        count_per_sample = torch.zeros(num_samples, device=loss.device, dtype=loss.dtype)

        if self.apply_mean:
            loss = loss.mean(self.mean_dim)
        loss_per_sample = loss_per_sample.scatter_add_(0, batch.type(torch.int64), loss)
        count_per_sample = count_per_sample.scatter_add_(0, batch.type(torch.int64), torch.ones(batch_size, device=loss.device, dtype=loss.dtype))

        # Compute mean loss for each sample
        loss_batch = loss_per_sample / count_per_sample

        # Return the overall mean of the mean losses
        return loss_batch

class GeometricCrossEntropy(GeometricLoss):

    def _loss(self, pred, targ):
        return F.cross_entropy(pred, targ, reduction='none')

class GeometricHuber(GeometricLoss):

    def _loss(self, pred, targ):
        return F.huber_loss(pred, targ, reduction='none')

class GeometricL2(GeometricLoss):

    def _loss(self, pred, targ):
        return 0.5 * F.mse_loss(pred, targ, reduction='none')

class GeometricL1(GeometricLoss):

    def _loss(self, pred, targ):
        return F.l1_loss(pred, targ, reduction='none')

class GeometricKL(GeometricLoss):

    def _loss(self, mu, log_var):
        kl_loss = -0.5 * (1 + log_var - mu**2 - log_var.exp())
        return kl_loss

GeometricLosses = {
    'l2': GeometricL2,
    'l1': GeometricL1,
    'huber': GeometricHuber,
    'cross_entropy': GeometricCrossEntropy,
    'kl': GeometricKL
}