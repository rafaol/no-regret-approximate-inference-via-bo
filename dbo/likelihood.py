from collections import OrderedDict
import numpy as np
import torch
import scipy.spatial.distance as spd
from dbo import simulation


def make_covariance_matrix(cov, n_dim=None):
    if isinstance(cov, torch.Tensor):
        cov = cov.cpu().numpy()
    if isinstance(cov, np.ndarray):
        if cov.ndim > 1:
            cov = cov
        else:
            cov = np.diag(cov)
    else:
        assert isinstance(cov, float), "Covariance parameter must at least be a floating point scalar"
        cov = np.eye(n_dim) * cov
    return cov


class ApproxTrajectoryLogLikelihood:
    def __init__(self, target_trajectory, noise_std=1):
        self.target_trajectory = target_trajectory
        self.noise_std = noise_std

    def __call__(self, trajectory):
        return torch.distributions.MultivariateNormal(trajectory.view(-1),
                                                      torch.eye(trajectory.numel()) * self.noise_std ** 2
                                                      ).log_prob(self.target_trajectory.view(-1))


def calculate_cross_correlation(states_trajectory: np.ndarray, actions_trajectory: np.ndarray):
    n_steps = len(states_trajectory)
    if len(actions_trajectory.shape) < 2:
        actions_trajectory = np.atleast_2d(actions_trajectory).T

    cur_state = states_trajectory[:-1]
    next_state = states_trajectory[1:]
    cur_action = actions_trajectory[:-1]
    sdim = cur_state.shape[1]
    adim = cur_action.shape[1]
    state_difference = next_state - cur_state
    # state_difference = np.array(cur_state)
    actions = np.array(cur_action)
    sample = np.zeros((sdim, adim))
    for i in range(sdim):
        for j in range(adim):
            sample[i, j] = np.dot(state_difference[:, i], actions[:, j]) / (n_steps - 1)
            # Add mean of absolut states changes and std to the summary statistics

    sample = sample.reshape(-1)
    sample = np.append(sample, np.mean(state_difference, axis=0))
    sample = np.append(sample, np.std(state_difference.astype(np.float64), axis=0))

    stats = np.array(sample)

    return stats


class ApproxTrajectoryAutocorrelationLogLikelihood:
    """
    Likelihood function for simulated trajectories based on auto-correlation summary statistics.
    """
    def __init__(self, target_obs_trajectory, actions_trajectory, cov=1e-2, dtype=None, device=None,
                 transform=True):
        self.dtype = dtype
        self.device = device
        self.target_stats = calculate_cross_correlation(target_obs_trajectory, actions_trajectory)
        self.actions_trajectory = actions_trajectory
        if isinstance(cov, torch.Tensor):
            cov = cov.cpu().numpy()
        if isinstance(cov, np.ndarray):
            if cov.ndim > 1:
                self.cov = cov
            else:
                self.cov = np.diag(cov)
        else:
            assert isinstance(cov, float), "Covariance parameter must at least be a floating point scalar"
            self.cov = np.eye(self.target_stats.size) * cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.transform = transform

    def __call__(self, trajectory):
        """
        Currently implements log "likelihood" as a discrepancy function using the negative Mahalanobis distance between
        the summary statistics of the simulated trajectory and the observed trajectory.
        :param trajectory: a observations trajectory from a RL simulator
        :return: a scalar tensor
        """
        if self.transform:
            stats = calculate_cross_correlation(trajectory, self.actions_trajectory)
        else:
            stats = trajectory

        return - torch.tensor(spd.mahalanobis(self.target_stats, stats, self.cov_inv),
                              dtype=self.dtype, device=self.device).pow(2).sum()


class TrajectorySetLogLikelihood:
    """
    Computes the likelihood function for a given parameter based on a set of independent observation and action
    trajectories.
    """
    def __init__(self, target_obs_trajectories, target_action_trajectories, env_id, param_names,
                 sigma=1e-2, dtype=None, device=None, squeeze_actions=True):
        """

        :param target_obs_trajectories: list of observation trajectories from the target environment
        :param target_action_trajectories: list of action trajectories applied to the target environment
        :param env_id: Gym environment ID
        :param param_names: list of parameter names, following the same order to be passed in the queries
        :param dtype: torch data type to enforce
        :param device: torch device to enforce
        :param squeeze_actions: whether or not to squeeze actions into 1D arrays, required by some environments.
        """
        self.env_id = env_id
        if len(target_action_trajectories[0].shape) < 2:
            self.action_trajectories = [traj[:, None] for traj in target_action_trajectories]
        else:
            self.action_trajectories = target_action_trajectories

        self.likelihood_fn = [ApproxTrajectoryAutocorrelationLogLikelihood(obs, actions, cov=sigma,
                                                                           dtype=dtype, device=device)
                              for obs, actions in zip(target_obs_trajectories, self.action_trajectories)]

        self.initial_obs = [obs[0] for obs in target_obs_trajectories]
        if squeeze_actions:
            self.action_trajectories = [traj.squeeze() for traj in self.action_trajectories]
        self.param_names = param_names
        obs_stats = []
        for target_obs_trajectory, actions_trajectory in zip(target_obs_trajectories, target_action_trajectories):
            obs_stats += [calculate_cross_correlation(target_obs_trajectory, actions_trajectory)]
        self.obs_stats = np.stack(obs_stats).ravel()
        self.sigma = sigma

    def __call__(self, sim_params):
        param_settings = OrderedDict(zip(self.param_names, sim_params.tolist()))
        sim = simulation.Simulator(self.env_id, param_settings)
        n = len(self.action_trajectories)
        # log_likelihood = torch.zeros(n, dtype=self.likelihood_fn[0].dtype, device=self.likelihood_fn[0].device)
        sim_stats = []
        for i in range(n):
            sim_trajectory = np.array(sim.run_actions(self.action_trajectories[i], initial_obs=self.initial_obs[i]))
            sim_stats += [calculate_cross_correlation(sim_trajectory, self.action_trajectories[i])]
        sim_stats = np.stack(sim_stats).ravel()
        discrepancy = np.linalg.norm((sim_stats - self.obs_stats)/self.sigma)
        return -torch.tensor(discrepancy, dtype=torch.get_default_dtype())
