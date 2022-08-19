from contextlib import contextmanager
import glob
import math
import os
import pickle
import re

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch


def cov(m: torch.Tensor, row_var: bool = False, inplace: bool = False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        row_var: If `row_var` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
        inplace: If `inplace` is True, it performs computation in place, i.e.
            modifying the input tensor.

    Returns:
        The covariance matrix of the variables.

    Source: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not row_var and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt)


def squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes squared distance matrix between two arrays of row vectors.

    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res


def estimate_lengthscale(x: torch.Tensor, y: torch.Tensor, ard: bool = False) -> torch.Tensor:
    if ard:
        diff = (x.t()[None, :, :] - y[:, :, None])
        h = torch.median(diff.transpose(-2, -1) ** 2, dim=1).values
        h = h.median(dim=0).values
    else:
        pairwise_dists = squared_distance(x, y)
        h = torch.median(pairwise_dists)
    h = torch.sqrt(0.5 * h / math.log(x.shape[0] + 1))
    return h


def estimate_rkhs_norm(fun, points: torch.Tensor, kernel: gpytorch.kernels.Kernel):
    f_values = fun(points).view(-1, 1)
    alpha = torch.solve(f_values, kernel(points).evaluate()).solution.view(-1)
    return alpha.dot(f_values.view(-1)).sqrt()


def make_grid(x_lb, x_ub, sx, y_lb=None, y_ub=None, sy=None, centred=True):
    if y_lb is None:
        y_lb = x_lb
    if y_ub is None:
        y_ub = x_ub
    if sy is None:
        sy = sx

    xs, ys = np.mgrid[x_lb:x_ub:sx, y_lb:y_ub:sy].astype('float')
    if centred:
        xs += 0.5 * sx
        ys += 0.5 * sy
    np_array = np.vstack((xs.ravel(), ys.ravel())).T

    return torch.tensor(np_array, dtype=torch.get_default_dtype())


@contextmanager
def cd(newdir: str):
    """
    Change current working directory.

    Original: https://stackoverflow.com/a/24176022

    :param newdir: path to new directory
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def save_object(filename, obj, protocol=None):
    """
    Save object using Pickle

    :param filename: file name
    :param obj: Python object
    :param protocol: pickle protocol to pass on to pickle.dump()
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def load_object(filename):
    """
    Loads pickled object

    :param filename: file with the object
    :return: the object
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_gp(filename: str):
    gp_model = load_object(filename).to(torch.empty([]))    # load GP model with tensor type converted to default
    gp_model.eval()
    gp_model.requires_grad_(False)
    gp_model.recalculate()
    return gp_model


def find_trajectory_files(trajectories_dir, env_type):
    """
    Finds NumPy files with observations and actions trajectories

    :param trajectories_dir: the directory to look at
    :param env_type: the environment type
    :type env_type: str
    :return: a tuple of two lists of strings, observations and actions trajectory file paths
    """
    obs_paths = glob.glob("{}/*{}-*-observations.npy".format(trajectories_dir, env_type))
    actions_paths = glob.glob("{}/*{}-*-actions.npy".format(trajectories_dir, env_type))
    return obs_paths, actions_paths


def load_trajectories(trajectories_dir, env_type):
    """
    Load RL trajectories from the given directory.

    :param trajectories_dir: directory to look at
    :param env_type: environment type
    :return: tuple with two lists of numpy arrays, observations and actions trajectories
    """
    obs_paths, actions_paths = find_trajectory_files(trajectories_dir, env_type)
    n_obs_traj = len(obs_paths)
    n_actions_traj = len(actions_paths)
    if n_obs_traj != n_actions_traj:
        raise RuntimeError(
            "Unequal number of observations and actions trajectories in directory {} for environment type {}".format(
                trajectories_dir, env_type))

    if n_obs_traj == 0:
        raise RuntimeError("No trajectories found at: {}".format(trajectories_dir))

    obs_traj = [None] * n_obs_traj
    actions_traj = [None] * n_actions_traj

    for obs_path in obs_paths:
        fname = os.path.basename(obs_path)
        i = int(re.search(r"\d+-observations.npy", fname).group().split("-", 1)[0]) - 1
        obs_traj[i] = np.load(obs_path)

    for actions_path in actions_paths:
        fname = os.path.basename(actions_path)
        i = int(re.search(r"\d+-actions.npy", fname).group().split("-", 1)[0]) - 1
        actions_traj[i] = np.load(actions_path)

    return obs_traj, actions_traj


def plot_estimates(model, lb, ub, observations=None, **kwargs):
    extent = [lb, ub, lb, ub]
    n_points = int(model.shape[0]**0.5)
    plt.imshow(model.view(n_points, n_points).t(), extent=extent, origin='lower', **kwargs)
    if observations is not None:
        plt.plot(observations[:, 0].cpu(), observations[:, 1].cpu(), 'k+')
    plt.axis(extent)
    plt.colorbar()


def load_abc_samples(directory="experiments", problem="CartPole"):
    abc_result = load_object(os.path.join(directory, f"ABC-result-{problem}-Gaussian-prior.pkl"))
    return torch.tensor(abc_result.samples_array, dtype=torch.get_default_dtype())


def configure_matplotlib(small_size=10, medium_size=12, bigger_size=14):
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    plt.rc('image', cmap='jet')
