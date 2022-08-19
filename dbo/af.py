"""
Acquisition functions
"""
import math
import torch
from torch.distributions import Distribution
from .gp import GPModel
import abc
from typing import Union


class BaseUCB(abc.ABC):
    """
    Base class for UCB methods
    """
    def __init__(self, delta):
        self._beta_t = torch.zeros([])
        self.delta = delta

    @abc.abstractmethod
    def ucb_parameter(self, delta):
        pass

    def update(self):
        """
        Update UCB parameter according to confidence level.
        """
        self._beta_t = self.ucb_parameter(self.delta)

    @property
    def beta_t(self) -> torch.Tensor:
        """
        UCB parameter
        """
        return self._beta_t

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()


class PlainUCB(BaseUCB):
    """
    Gaussian process upper confidence bound (GP-UCB)
    """
    def __init__(self, gp_model: GPModel,
                 f_bound: Union[float, torch.Tensor],
                 sigma_out: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """
        Constructor.

        :param gp_model: a `GPModel` for the objective function
        :param f_bound: upper bound on the RKHS norm of the objective function
        :param sigma_out: sub-Gaussian parameter for the output noise
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-`delta` (Default: 0.1)
        """
        super().__init__(delta)
        self.gp_model = gp_model
        self.f_bound = f_bound
        self.sigma_out = sigma_out
        self._beta_t = self.ucb_parameter(delta)

    @property
    def name(self):
        return "GP-UCB"

    def ucb_parameter(self, delta):
        ig = self.gp_model.information_gain()
        lam = self.gp_model.likelihood.noise
        beta_kt = self.f_bound + self.sigma_out * torch.sqrt(2 * (ig - math.log(delta))/lam)
        return beta_kt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.gp_model(x)
        return pred.mean + self.beta_t * pred.variance.sqrt()


class PointDUCB(PlainUCB):
    def __init__(self, prior: Distribution, gp_model: GPModel,
                 f_bound: Union[float, torch.Tensor],
                 sigma_out: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """
        Constructor.

        :param gp_model: a `GPModel` for the objective function
        :param f_bound: upper bound on the RKHS norm of the objective function
        :param sigma_out: sub-Gaussian parameter for the output noise
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-`delta` (Default: 0.1)
        """
        super().__init__(gp_model, f_bound, sigma_out, delta)
        self.prior = prior

    @property
    def name(self):
        return "GP-UCB"

    def ucb_parameter(self, delta):
        ig = self.gp_model.information_gain()
        lam = self.gp_model.likelihood.noise
        beta_kt = self.f_bound + self.sigma_out * torch.sqrt(2 * (ig - math.log(delta)) / lam)
        return beta_kt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.gp_model(x)

        return pred.variance * (2 * pred.mean + self.beta_t * pred.variance.sqrt())
