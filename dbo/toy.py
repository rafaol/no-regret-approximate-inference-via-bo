import gpytorch
import torch
from torch.distributions import Distribution, Normal, MultivariateNormal, Independent, MixtureSameFamily, Categorical


class GaussianObjective:
    def __init__(self, n_dim=1, **kwargs):
        self.dist = MultivariateNormal(loc=4 * torch.ones(n_dim), scale_tril=torch.eye(n_dim))

    def __call__(self, x: torch.Tensor):
        return self.dist.log_prob(x)


class MixtureObjective:
    def __init__(self, n_dim: int = 1, n_components: int = 12, **kwargs):
        self.means = torch.rand(n_components, n_dim)
        self.scales = torch.rand(n_components, n_dim) * 0.4 + 0.2
        self.n_components = n_components
        components = Independent(Normal(self.means, scale=self.scales), 1)
        self.dist = MixtureSameFamily(Categorical(torch.ones(n_components)), components)

    def __call__(self, x: torch.Tensor):
        return self.dist.log_prob(x)

    def posterior(self, prior: MultivariateNormal):
        prior_scale = prior.covariance_matrix.diag().sqrt()
        means = (self.means * prior_scale + prior.mean * self.scales) / (prior_scale + self.scales)
        scales = prior_scale * self.scales / (prior_scale + self.scales)
        components = Independent(Normal(means, scale=scales), 1)
        dist = MixtureSameFamily(Categorical(torch.ones(self.n_components)), components)
        return dist


class RKHSObjective:
    def __init__(self, kernel: gpytorch.kernels.Kernel, prior: Distribution = None, n_points: int = 10, points=None,
                 weights=None, **kwargs):
        if points is None:
            points = prior.sample(torch.Size([n_points]))
        self.points = points
        if weights is None:
            weights = MultivariateNormal(torch.zeros(n_points), scale_tril=torch.eye(n_points)).sample().view(-1, 1)
        self.weights = weights

        self.kernel = kernel
        k_matrix = self.kernel(self.points).evaluate()
        self._norm = (self.weights.t() @ k_matrix @ self.weights).sqrt().view([])

    def __setstate__(self, state):
        self.__init__(state['kernel'](), points=state['points'], weights=state['weights'])
        self.kernel.load_state_dict(state['kernel_state'])

    def __getstate__(self):
        state = dict()
        state['points'] = self.points
        state['weights'] = self.weights
        state['kernel'] = self.kernel.__class__
        state['kernel_state'] = self.kernel.state_dict()
        return state

    @property
    def norm(self):
        return self._norm

    def __call__(self, x: torch.Tensor):
        return (self.kernel(x, self.points) @ self.weights).view(x.shape[0])


class CircularObjective:
    def __init__(self, n_dim: int = 2, lenghtscale: float = 0.25, **kwargs):
        self.centre = torch.zeros(1, n_dim)
        self.radius = torch.ones([]) * 1.5
        self.lengthscale = lenghtscale

    def __call__(self, x: torch.Tensor):
        dists = torch.norm(x - self.centre, dim=1, keepdim=True)
        cost = torch.norm((dists - self.radius) / self.lengthscale, dim=1) ** 2
        return -cost
