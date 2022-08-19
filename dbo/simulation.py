import gym

import numpy as np
import random
import math


def is_gym_env(env_id):
    try:
        gym.spec(env_id)
        check = True
    except gym.error.Error:
        check = False
    return check


def make_env(env_id, rank, param_settings=None, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param param_settings (dict): dictionary of parameter names with the corresponding array of values
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        if param_settings is not None:
            uw_env = env.unwrapped
            env_type = env_id.split('-', 1)[0]
            print("{} env. {}:".format(env_type, rank + 1))
            for param in param_settings:
                value = param_settings[param][rank]
                setattr(uw_env, param, value)
                print("{}: {}".format(param, value))
            else:
                print("Unknown Gym environment to randomise!")
        return env

    return _init


def run_episode(model, env, max_steps=1000, seed=None, render=False):
    """
    Run RL agent for a single episode.
    
    :param render: render animation
    :param model: (BaseRLModel object) the RL Agent
    :param env: Gym environment
    :param max_steps: (int) maximum number of timesteps to evaluate it
    :param seed: (int or None) if not None, reseed the environment with seed
    :return: (float) reward
    """
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    episode_reward = 0
    done = False
    t = 0
    obs_trajectory = [None] * max_steps
    actions_trajectory = [None] * max_steps
    _states = None
    while not done and t < max_steps:
        obs_trajectory[t] = obs
        if render:
            env.render()
        actions, _states = model.predict(obs, state=_states, mask=done, deterministic=True)
        actions_trajectory[t] = actions
        obs, reward, done, info = env.step(actions)
        episode_reward += reward
        t += 1
    actions_trajectory = actions_trajectory[:t]
    obs_trajectory = obs_trajectory[:t]
    return episode_reward, obs_trajectory, actions_trajectory


def get_state_from_obs(obs, env_type):
    if env_type == "Pendulum":
        return np.array([math.atan2(obs[1], obs[0]), obs[2]])
    return obs


def get_env_type(env_id):
    return env_id.split('-', 1)[0]


def get_env_id(env_type):
    return "{}-v0".format(env_type)


class Simulator(object):
    """
    Gym environment simulator.
    """
    def __init__(self, env_id, param_settings, reseed=False, seed=0):
        """
        Simulator constructor.

        :param env_id: Gym environment ID (str)
        :param param_settings: dictionary whose keys are parameter names and values are the corresponding setting
        :type param_settings: dict
        """
        self.param_settings = param_settings
        self.env_id = env_id
        self.env_type = get_env_type(self.env_id)
        self.env = gym.make(self.env_id)
        self.env.seed(seed)
        self.reseed = reseed
        # param_names = randomising[self.env_type]
        for p in param_settings:
            setattr(self.env.unwrapped, p, param_settings[p])

    def __del__(self):
        try:
            self.env.close()
        except ConnectionError as ex:
            return

    def run_actions(self, actions_trajectory, initial_state=None, initial_obs=None):
        """
        Runs a sequence of actions on the simulator.

        :param actions_trajectory: list of actions, which should be valid for the environment
        :param initial_state: optional initial state to set the environment at
        :param initial_obs: optional initial observation, which is used to infer the initial state
        :return: the resulting sequence of observations
        :rtype: list
        """
        obs = self.env.reset()
        if initial_state is not None:
            self.env.unwrapped.state = initial_state
            obs = initial_state
        if initial_obs is not None:
            self.env.unwrapped.state = get_state_from_obs(initial_obs, self.env_type)
            obs = initial_obs

        n_steps = len(actions_trajectory)
        obs_trajectory = [None] * n_steps
        for t in range(n_steps):
            obs_trajectory[t] = obs
            actions = actions_trajectory[t]
            try:
                obs, _, done, _ = self.env.step(actions)
            except ValueError as ex:
                print("Error in simulation step:")
                print("actions applied:", actions)
                print("previous obs:", obs)
                print("param settings:", self.param_settings)
                raise ex
            if done:
                break

        for i, o in enumerate(obs_trajectory):
            if o is None:
                obs_trajectory[i] = obs
        return obs_trajectory

    def __call__(self, policy, max_steps=1000):
        if self.reseed:
            seed = random.getrandbits(32)
            return run_episode(policy, self.env, max_steps, seed=seed)

        return run_episode(policy, self.env)
