from typing import Callable
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from .sampling import density_estimator
from .util import cov
import ite


def gs_divergence(reference_samples: torch.Tensor, samples: torch.Tensor):
    p = MultivariateNormal(reference_samples.mean(dim=0),
                           covariance_matrix=cov(reference_samples, row_var=False))
    q = MultivariateNormal(samples.mean(dim=0),
                           covariance_matrix=cov(samples, row_var=False))   # FIXED w.r.t. the original paper code
    return 0.5*(kl_divergence(p, q) + kl_divergence(q, p))


def sample_divergence(reference_samples: torch.Tensor, samples: torch.Tensor):
    de = density_estimator(samples)
    kl_estimator = ite.cost.BDKL_KnnK()
    kl = kl_estimator.estimation(de.sample([samples.shape[0]]).cpu().numpy(),
                                 reference_samples.cpu().numpy())
    return kl


def evidence_lower_bound(log_posterior: Callable[[torch.Tensor], torch.Tensor], samples: torch.Tensor):
    de = density_estimator(samples)
    de_samples = de.sample([samples.shape[0]])
    elbo = (log_posterior(de_samples) - de.log_prob(de_samples)).mean()  # negative cross entropy
    assert not torch.isnan(elbo)
    return elbo
