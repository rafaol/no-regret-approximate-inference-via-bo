import math

import emcee
import numpy as np
import pyro
from pyro.distributions import TorchDistribution
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model, init_to_uniform
from pyro.distributions.util import scalar_like
from pyro.util import torch_isnan

from KDEpy import bw_selection

import torch
from torch.distributions import constraints, MixtureSameFamily, Normal, Independent, Categorical,\
    Distribution, Uniform

from scipy.stats import gaussian_kde


def density_estimator(samples) -> MixtureSameFamily:
    nd = samples.shape[1]
    bandwidths = np.hstack([bw_selection.scotts_rule(samples[:, i, None].cpu().numpy())
                            for i in range(nd)])
    bandwidths = torch.tensor(bandwidths, dtype=torch.get_default_dtype())
    components = Independent(Normal(samples, bandwidths), 1)
    n = samples.shape[0]
    estimator = MixtureSameFamily(Categorical(torch.ones(n)), components)
    return estimator


class ScipyKDE(Distribution):
    def __init__(self, samples: torch.Tensor):
        super(ScipyKDE, self).__init__()
        self._samples = samples
        np_samples = samples.cpu().t().numpy()
        self.kde = gaussian_kde(np_samples)
        self._variance = self._samples.var()
        self._mean = self._samples.mean(dim=0)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def log_prob(self, x: torch.Tensor):
        return torch.tensor(self.kde.logpdf(x.cpu().t().numpy())).t().view(x.shape[0])

    def sample(self, sample_shape=torch.Size()):
        return torch.tensor(self.kde.resample(sample_shape[0])).t()


class SimpleDensityEstimator(Distribution):
    def __init__(self, log_likelihood, prior: Distribution, n_samples: int = 1000):
        super(SimpleDensityEstimator, self).__init__()
        self.log_likelihood = log_likelihood
        self.prior = prior
        samples = prior.sample(torch.Size([n_samples]))
        self.log_evidence = torch.logsumexp(log_likelihood(samples), dim=0) - math.log(n_samples)

    def log_prob(self, x: torch.Tensor):
        return self.log_likelihood(x) + self.prior.log_prob(x) - self.log_evidence


class UnnormalisedDistribution(TorchDistribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, log_likelihood, prior: Distribution):
        super(TorchDistribution, self).__init__()
        self.log_likelihood = log_likelihood
        self.prior = prior
        self.n_dim = prior.mean.shape[-1]

    # HACK: Only used for model initialization.
    def sample(self, sample_shape=torch.Size()):
        return self.prior.sample(sample_shape)

    def log_prob(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        lik_logp = self.log_likelihood(x.view(-1, self.n_dim))
        prior_logp = self.prior.log_prob(x).view(lik_logp.shape[0], -1)
        prior_logp = prior_logp.sum(dim=1)
        return prior_logp + lik_logp

    def emcee_log_prob(self, x: np.ndarray):
        lp = self.log_prob(torch.tensor(x, dtype=torch.get_default_dtype()))
        return lp.numpy()


class MCMCSampler:
    def __init__(self, likelihood, prior: Distribution, n_chains: int = 1, use_jit: bool = False,
                 sampler: str = 'NUTS'):
        self.un_dist = UnnormalisedDistribution(likelihood, prior)
        self._last_samples = None
        self._n_chains = n_chains
        self.use_jit = use_jit
        self.sampler = sampler

    def model(self):
        pyro.sample("x", self.un_dist)

    @property
    def last_samples(self):
        return self._last_samples

    def sample(self, n: int, factor=1, n_burn=100, initial_param: torch.Tensor = None,
               n_emcee_walkers: int = 25) -> torch.Tensor:
        if self.sampler in ['NUTS', 'RW']:
            if self.sampler == 'NUTS':
                kernel = NUTS(self.model, jit_compile=self.use_jit, ignore_jit_warnings=self.use_jit,
                              init_strategy=init_to_uniform)
            elif self.sampler == 'RW':
                kernel = RWKernel(self.model, jit_compile=self.use_jit, ignore_jit_warnings=self.use_jit,
                                  init_strategy=init_to_sample, step_size=0.2)
            mcmc = MCMC(kernel=kernel, num_samples=n*factor//self._n_chains, warmup_steps=n_burn,
                        num_chains=self._n_chains, mp_context="spawn", disable_progbar=(self._n_chains > 1),
                        initial_params={'x': initial_param} if initial_param is not None else None)
            mcmc.run()
            self._last_samples = mcmc.get_samples()['x']
            return mcmc.get_samples(n)['x']
        elif self.sampler == 'EMCEE':
            if n_emcee_walkers < 2*self.un_dist.n_dim:
                n_emcee_walkers *= 2
            sampler = emcee.EnsembleSampler(nwalkers=n_emcee_walkers, ndim=self.un_dist.n_dim,
                                            log_prob_fn=self.un_dist.emcee_log_prob, vectorize=True)
            if initial_param is None:
                initial_state = self.un_dist.sample(torch.Size([n_emcee_walkers]))
            else:
                if initial_param.shape == (n_emcee_walkers, self.un_dist.n_dim):
                    initial_state = initial_param
                else:
                    initial_state = initial_param.expand(n_emcee_walkers, self.un_dist.n_dim)
            state = sampler.run_mcmc(initial_state.numpy(), n_burn, progress=True)
            sampler.reset()
            sampler.run_mcmc(state, n*factor//n_emcee_walkers, progress=True)
            samples = sampler.get_chain(flat=True)
            self._last_samples = torch.tensor(samples, dtype=torch.get_default_dtype())
            return self._last_samples.index_select(0, torch.randint(samples.shape[0], size=[n]))
        else:
            raise RuntimeError("Unrecognised MCMC sampler type")

    def divergence(self, log_p, n: int = 1000, samples: torch.Tensor = None):
        if samples is None:
            samples = self.sample(n)
        de = density_estimator(samples)
        de_samples = samples
        cross_entropy = -log_p(de_samples).mean()
        kl = cross_entropy + de.log_prob(de_samples).mean()
        return kl


class RWKernel(MCMCKernel):
    def __init__(self,
                 model=None,
                 step_size: float = 1,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 init_strategy=init_to_uniform,
                 initial_params = None):
        super(RWKernel, self).__init__()
        if not torch.is_tensor(step_size):
            step_size = torch.tensor(step_size, dtype=torch.get_default_dtype())
        self._step_size = step_size
        self.model = model
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings
        self._init_strategy = init_strategy
        self._max_plate_nesting = max_plate_nesting
        self._initial_params = None
        self.transforms = None
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = initial_params
        self._z_last = None
        self._potential_energy_last = None
        self._warmup_steps = None

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.
        self._divergences = []
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._potential_energy_last = None
        self._warmup_steps = None

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
            init_strategy=self._init_strategy,
            initial_params=self._initial_params,
            )
        self.transforms = transforms
        self._initial_params = init_params
        self._prototype_trace = trace
        self.potential_fn = potential_fn

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        """
        Sets the parameters to initiate the MCMC run. Note that the parameters must
        have unconstrained support.
        """
        self._initial_params = params

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            potential_energy = self.potential_fn(z)
        else:
            potential_energy = self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy)

    def cleanup(self):
        self._reset()

    def _cache(self, z, potential_energy):
        self._z_last = z
        self._potential_energy_last = potential_energy

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last

    def sample(self, params):
        z, potential_energy = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            potential_energy = self.potential_fn(z)
            self._cache(z, potential_energy)
        # return early if no sample sites
        elif len(z) == 0:
            self._t += 1
            self._mean_accept_prob = 1.
            if self._t > self._warmup_steps:
                self._accept_cnt += 1
            return params

        z_new = z.copy()
        for k, v in z_new.items():
            z_new[k] = v + torch.randn_like(v)*self._step_size
        potential_energy_new = self.potential_fn(z_new)

        # apply Metropolis correction.
        delta_energy = potential_energy_new - potential_energy

        # handle the NaN case which may be the case for a diverging trajectory
        # when using a large step size.
        delta_energy = scalar_like(delta_energy, float("inf")) if torch_isnan(delta_energy) else delta_energy
        if self._t >= self._warmup_steps:
            self._divergences.append(self._t - self._warmup_steps)

        accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = Uniform(scalar_like(accept_prob, 0.), scalar_like(accept_prob, 1.)).sample([])

        accepted = False
        if rand < accept_prob:
            accepted = True
            z = z_new
            self._cache(z, potential_energy_new)

        self._t += 1
        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t

        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n
        return z.copy()

    def diagnostics(self):
        return {"divergences": self._divergences,
                "acceptance rate": self._accept_cnt / (self._t - self._warmup_steps)}
