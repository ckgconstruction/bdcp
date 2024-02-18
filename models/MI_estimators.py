import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

SMALL = 1e-08


class UpperBound(nn.Module, ABC):
    @abstractmethod
    def update(self, y_samples):
        raise NotImplementedError


class UpperWithPosterior(UpperBound):

    def __init__(
            self,
            embedding_dim=768,
            hidden_dim=500,
            tag_dim=128,
            device=None
    ):
        super(UpperWithPosterior, self).__init__()
        self.device = device

        self.p_mu = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),

                                  nn.Tanh(),
                                  nn.Linear(hidden_dim, tag_dim))

        self.p_log_var = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                       nn.Tanh(),

                                       nn.Linear(hidden_dim, tag_dim),

                                       )

    def get_mu_logvar(self, embeds):
        mean = self.p_mu(embeds)
        log_var = self.p_log_var(embeds)

        return mean, log_var

    def loglikeli(self, y_samples, mu, log_var):
        return (-0.5 * (mu - y_samples) ** 2 / log_var.exp()
                + log_var
                + torch.log(math.pi)
                ).sum(dim=1).mean(dim=0)

    def get_sample_from_param_batch(self, mean, log_var, sample_size):
        bsz, seqlen, tag_dim = mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to(self.device)

        z = z * torch.exp(0.5 * log_var).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)

        return z

    @abstractmethod
    def update(self, y_samples):
        raise NotImplementedError


class VIB(UpperWithPosterior):

    def update(self, x_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()


class CLUB(UpperWithPosterior):

    def mi_est_sample(self, y_samples, mu, log_var):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = - (mu - y_samples) ** 2 / 2. / log_var.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / 2. / log_var.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return upper_bound

    def update(self, x_samples):
        mu, log_var = self.get_mu_logvar(x_samples)
        y_samples = self.get_sample_from_param_batch(mu, log_var, 1).squeeze(0)

        return self.mi_est_sample(y_samples, mu, log_var)


class vCLUB(UpperBound):

    def __init__(self):
        super(vCLUB, self).__init__()

    def mi_est_sample(self, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        return self.mi_est(y_samples, y_samples[random_index])

    def mi_est(self, y_samples, y_n_samples):
        positive = torch.zeros_like(y_samples)

        negative = - (y_samples - y_n_samples) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return upper_bound

    def mse(self, y_samples, y_n_samples):
        return (y_samples - y_n_samples) ** 2 / 2

    def consine(self, y_samples, y_n_samples):
        return torch.cosine_similarity(y_samples, y_n_samples, dim=-1)

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, y_samples, y_n_samples):
        return self.mi_est(y_samples, y_n_samples)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, device=None):
        super(InfoNCE, self).__init__()
        self.lower_size = 100
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                    nn.ReLU(),
                                    nn.Linear(self.lower_size, 1),
                                    nn.Softplus())
        self.device = device

    def forward(self, x_samples, y_samples):

        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        try:
            T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))
        except Exception:
            print('Debug')

        lower_bound = T0 - T1.logsumexp(dim=1)

        return lower_bound.mean() * -1

    def span_mi_loss(self, x_spans, x_span_idxes, y_spans, y_span_idxes):

        bsz, sample_size, _, dim = x_spans.shape
        x_span_idxes = x_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)
        y_span_idxes = y_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)

        try:
            x_spans_con = x_spans[:, 0, :, :].squeeze(axis=1).gather(1, x_span_idxes).squeeze(1)
            y_spans_con = y_spans[:, 0, :, :].squeeze(axis=1).gather(1, y_span_idxes).squeeze(1)
        except Exception:
            print('Error!')

        info_nce_loss = self.forward(x_spans_con, y_spans_con)

        return info_nce_loss * -1


def kl_div(param1, param2):
    mean1, log_cov1 = param1
    mean2, log_cov2 = param2
    cov1 = log_cov1.exp()
    cov2 = log_cov2.exp()
    bsz, seqlen, tag_dim = mean1.shape
    var_len = tag_dim * seqlen

    cov2_inv = 1 / cov2
    mean_diff = mean1 - mean2

    mean_diff = mean_diff.view(bsz, -1)
    cov1 = cov1.view(bsz, -1)
    cov2 = cov2.view(bsz, -1)
    cov2_inv = cov2_inv.view(bsz, -1)

    temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
    KL = 0.5 * (
            torch.sum(torch.log(cov2), dim=1)
            - torch.sum(torch.log(cov1), dim=1)
            - var_len
            + torch.sum(cov2_inv * cov1, dim=1)
            + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz)
    )

    return KL.mean()


def kl_norm(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1).mean()
