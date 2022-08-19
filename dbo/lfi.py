from collections import OrderedDict
import numpy as np
import torch

from . import simulation, likelihood
from .util import load_trajectories


class RLSimulatorModel:
    def __init__(self, target_obs_trajectories, target_action_trajectories, env_id, param_names, squeeze_actions=True):
        self.env_id = env_id
        if len(target_action_trajectories[0].shape) < 2:
            self.action_trajectories = [traj[:, None] for traj in target_action_trajectories]
        else:
            self.action_trajectories = target_action_trajectories

        self.initial_obs = [obs[0] for obs in target_obs_trajectories]
        self.param_names = param_names
        self.squeeze_actions = squeeze_actions

    def __call__(self, sim_params, batch_size=1, random_state=None):
        param_settings = OrderedDict(zip(self.param_names, np.atleast_1d(sim_params.squeeze()).tolist()))
        sim = simulation.Simulator(self.env_id, param_settings)
        n = len(self.action_trajectories)
        stats = []
        for i in range(n):
            actions = self.action_trajectories[i]
            if self.squeeze_actions:
                actions = actions.squeeze()
            sim_trajectory = np.array(sim.run_actions(actions, initial_obs=self.initial_obs[i]))
            stats += [likelihood.calculate_cross_correlation(sim_trajectory, self.action_trajectories[i])]
        out_stats = np.stack(stats).reshape(1, -1)
        assert not np.isnan(out_stats).any()
        return out_stats


class SimWrapper:
    """
    Wrapper for BOLFI
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, mass, length, batch_size=1, random_state=None):
        return self.model(np.stack([mass, length], axis=1), batch_size=batch_size, random_state=random_state)


class RLObjective:
    def __init__(self, n_dim: int = 2, likelihood_sigma=1e-1, env_id="CartPole-v0",
                 trajectories_dir="experiments/trajectories/", **kwargs):
        assert n_dim == 2
        self.n_dim = n_dim
        if simulation.is_gym_env(env_id):
            env_type = simulation.get_env_type(env_id)
            real_obs_trajectories, real_actions_trajectories = load_trajectories(trajectories_dir, env_type)
            n_real_trajectories = len(real_obs_trajectories)
            assert n_real_trajectories > 0

            if env_type == "CartPole":
                for i in range(n_real_trajectories):
                    real_actions_trajectories[i] = np.array(real_actions_trajectories[i], dtype=np.int)

            squeeze_actions = env_type in ["CartPole"]

            self.log_like = likelihood.TrajectorySetLogLikelihood(real_obs_trajectories,
                                                                  real_actions_trajectories,
                                                                  param_names=["masspole", "length"],
                                                                  env_id=env_id,
                                                                  sigma=likelihood_sigma,
                                                                  squeeze_actions=squeeze_actions)
        else:
            raise RuntimeError("Unrecognised environment '{}'".format(env_id))

    def __call__(self, x: torch.Tensor):
        assert x.numel() == self.n_dim
        return self.log_like(x.view(-1))


def id_summary(y):
    return y


def data_distance(x, y, sigma=1e-1):
    d = np.linalg.norm((x-y)/sigma, axis=1, keepdims=True)
    return d
