import argparse
import os
import random
import time
import yaml

import gpytorch
import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal

from dbo.af import PlainUCB
from dbo.gp import GPModel, LogProbMean
from dbo.sampling import MCMCSampler, density_estimator, UnnormalisedDistribution
import dbo.toy
from dbo.util import cd, make_grid, save_object, cov
from dbo.diagnostics import sample_divergence, gs_divergence, evidence_lower_bound


def plot_estimates(model, lb, ub, observations=None, **kwargs):
    extent = [lb, ub, lb, ub]
    n_points = int(model.shape[0]**0.5)
    plt.imshow(model.view(n_points, n_points).t(), extent=extent, origin='lower', **kwargs)
    if observations is not None:
        plt.plot(observations[:, 0].cpu(), observations[:, 1].cpu(), 'k+')
    plt.axis(extent)
    plt.colorbar()


def run_experiment(dimensionality, n_it, n_samples_per_it, n_chains, noise_level, objective_class,
                   output_directory, n_test=None, n_burn=400, delta=0.1, show_plots=False):

    problem_prior = MultivariateNormal(loc=torch.zeros(dimensionality), scale_tril=torch.eye(dimensionality))

    kernel = gpytorch.kernels.RBFKernel()
    if objective_class == 'CircularObjective':
        kernel.lengthscale = .5
    else:
        kernel.lengthscale = 0.5*dimensionality**0.5
    kernel.eval()
    kernel.requires_grad_(False)

    if n_test is None:
        if objective_class == 'CircularObjective':
            n_test = 10000
        else:
            if dimensionality <= 4:
                n_test = 1000
            else:
                n_test = dimensionality*2000

    gp_model = GPModel(gpytorch.kernels.ScaleKernel(kernel), mean_module=LogProbMean(problem_prior))

    if objective_class == 'RKHSObjective':
        objective = dbo.toy.RKHSObjective(n_dim=dimensionality, kernel=kernel, prior=problem_prior,
                                          n_points=10*dimensionality**2)
    else:
        objective = getattr(dbo.toy, objective_class)(n_dim=dimensionality, kernel=kernel,
                                                      prior=problem_prior)
    gp_model.covar_module.outputscale = 1
    if isinstance(objective, dbo.toy.RKHSObjective):
        f_norm = objective.norm
    else:
        # f_norm = objective(problem_prior.sample([n_test])).abs().max()
        # f_norm = estimate_rkhs_norm(objective, problem_prior.sample([n_test]), gp_model.covar_module)
        f_norm = 3

    if isinstance(objective, dbo.toy.MixtureObjective):
        true_posterior = objective.posterior(problem_prior)
        objective_samples = true_posterior.sample([n_test])
    else:
        objective_sampler = MCMCSampler(objective, problem_prior, n_chains=1, sampler='EMCEE')
        objective_samples = objective_sampler.sample(n_test, n_burn=n_burn)
        true_posterior = density_estimator(objective_samples)

    print(f"Objective moments:\nMean: {objective_samples.mean(dim=0)}\nCovariance:{cov(objective_samples)}")

    noise_sd = f_norm * noise_level
    if objective_class == 'RKHSObjective':
        gp_model.likelihood.noise = 1e-2
    else:
        gp_model.likelihood.noise = 1e-4  # torch.tensor(noise_sd ** 2)
    acq_fun = PlainUCB(gp_model, f_bound=f_norm, sigma_out=noise_sd, delta=delta)
    af_sampler = MCMCSampler(acq_fun, problem_prior, n_chains=n_chains, use_jit=True, sampler='EMCEE')

    n_points = 100
    x_lb = problem_prior.mean.mean().item() - 3 * problem_prior.variance.mean().sqrt().item()
    x_ub = problem_prior.mean.mean().item() + 3 * problem_prior.variance.mean().sqrt().item()

    f_test = None
    x_test = None
    if dimensionality == 2:
        x_test = make_grid(x_lb, x_ub, (x_ub - x_lb) / n_points)
        f_test = objective(x_test) + problem_prior.log_prob(x_test)

    divergences = torch.zeros(n_it)
    gs_divergences = torch.zeros(n_it)
    regret_bound = torch.zeros(n_it)
    elbo = torch.zeros(n_it)

    # BO loop
    for t in range(n_it):
        print(f"Sampling... {t + 1}/{n_it}")
        x_t = af_sampler.sample(n_samples_per_it, factor=100, n_burn=n_burn).view(n_samples_per_it, dimensionality)
        y_t = objective(x_t).view([n_samples_per_it]) + torch.randn([n_samples_per_it]) * noise_sd

        # Debug
        print("Computing divergence...")
        test_samples = af_sampler.last_samples
        elbo[t] = evidence_lower_bound(UnnormalisedDistribution(objective, problem_prior).log_prob, test_samples)
        gs_divergences[t] = gs_divergence(objective_samples, test_samples)
        divergences[t] = sample_divergence(objective_samples, test_samples)
        regret_bound[t] = 2 * acq_fun.beta_t * gp_model(test_samples).variance.sqrt().mean()
        print(f"Divergence/bound: {divergences[t]} < {regret_bound[t]}")
        print(f"gsKL divergence: {gs_divergences[t]}")
        print(f"ELBO: {elbo[t]}")

        if dimensionality == 2 and show_plots:
            plt.figure(figsize=(4, 8))
            af_test = af_sampler.un_dist.log_prob(x_test).cpu()
            plt.subplot(211)
            plot_estimates(f_test, x_lb, x_ub, vmin=-20, vmax=0)
            plt.subplot(212)
            plot_estimates(af_test, x_lb, x_ub, vmin=None, vmax=None)
            plt.plot(test_samples[:, 0], test_samples[:, 1], 'm+')
            plt.plot(x_t[:, 0].cpu(), x_t[:, 1].cpu(), 'k+', ms=10, mew=2)
            plt.show()

        # Update models
        gp_model.update(x_t, y_t)
        acq_fun.update()

    gp_sampler = MCMCSampler(gp_model.mean, problem_prior, n_chains=n_chains, use_jit=True, sampler='EMCEE')
    final_samples = gp_sampler.sample(n_test, n_burn=n_burn)
    final_de = density_estimator(final_samples)

    if dimensionality == 2 and show_plots:
        m_test = final_de.log_prob(x_test).exp().cpu()
        r_test = true_posterior.log_prob(x_test).exp().cpu()
        v_max = max(m_test.max().item(), r_test.max().item())
        plt.figure(figsize=(4, 7))
        plt.subplot(211)
        plot_estimates(r_test, x_lb, x_ub, vmin=0, vmax=v_max)
        plt.title("Target density")
        plt.subplot(212)
        plot_estimates(m_test, x_lb, x_ub, vmin=0, vmax=v_max)
        plt.title("Model density")
        plt.show()

    t_steps = torch.arange(n_it).cpu() + 1
    plt.figure(figsize=(4, 3))
    plt.plot(t_steps * n_samples_per_it, regret_bound.cumsum(-1).cpu() / t_steps, 'b-', label="Regret bound")
    plt.plot(t_steps * n_samples_per_it, divergences.cumsum(-1).cpu() / t_steps, 'k-', label="Averaged KL")
    plt.plot(t_steps * n_samples_per_it, divergences.cpu(), 'k+', label="KL divergence")
    plt.plot(t_steps * n_samples_per_it, gs_divergences.cumsum(-1).cpu() / t_steps, 'g-', label="Averaged gsKL")
    plt.plot(t_steps * n_samples_per_it, gs_divergences.cpu(), 'g+', label="gsKL divergence")
    plt.plot(t_steps * n_samples_per_it, elbo.cumsum(-1).cpu() / t_steps, 'r-', label="Averaged ELBO")
    plt.plot(t_steps * n_samples_per_it, elbo.cpu(), 'r+', label="ELBO")
    plt.xlabel("Evaluations")
    plt.legend()
    plt.tight_layout()
    with cd(output_directory):
        plt.savefig("result.png", dpi=300)
        torch.save(regret_bound, "dbo-bounds.pth")
        torch.save(divergences, "dbo-divergences.pth")
        torch.save(gs_divergences, "dbo-gs_divergences.pth")
        torch.save(elbo, "dbo-elbo.pth")
        save_object("dbo-gp.pkl", gp_model)
        save_object("dbo-objective.pkl", objective)
    if show_plots:
        plt.show()


def configure_matplotlib(small_size=10, medium_size=12, bigger_size=14):
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    plt.rc('image', cmap='jet')


def main(args):
    if args.output_directory is None:
        args.output_directory = os.path.join("experiments", time.strftime("toy-%Y-%m-%d-%H%M%S"))

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.seed is None:
        seed = random.SystemRandom().getrandbits(32)
    else:
        seed = args.seed
    torch.manual_seed(seed)

    output_directory = args.output_directory
    with cd(output_directory):
        with open("toy-seed.dat", "w") as f:
            f.write("{}\n".format(seed))
        with open("toy-args.yaml", "w") as f:
            yaml.dump(args, f)

    dim = args.dimensionality
    n_mcmc_chains = 1
    n_samples_per_iteration = args.n_samples_per_iteration
    n_iterations = args.n_iterations
    n_repeats = args.n_repeats
    objective_type = args.objective
    delta = 0.2
    if objective_type == 'RKHSObjective':
        log_lik_noise_level = 0.01
    else:
        log_lik_noise_level = 1e-6

    if dim == 2:
        configure_matplotlib()

    for r in range(n_repeats):
        if n_repeats > 1:
            print("##########################################")
            print(f"Running trial {r+1} out of {n_repeats}...")
            print("##########################################\n\n")
            run_dir = os.path.join(output_directory, f"run-{r+1:02}")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
        else:
            run_dir = output_directory
        run_experiment(dim, n_iterations, n_samples_per_iteration, n_mcmc_chains, log_lik_noise_level,
                       objective_type, output_directory=run_dir, delta=delta, show_plots=args.show_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy experiment with KL-UCB")
    parser.add_argument("-r", "--n-repeats", help="Number of repetitions", type=int, default=1)
    parser.add_argument("-t", "--n-iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-d", "--dimensionality", help="Dimensionality of the problem", type=int, default=2)
    parser.add_argument("-o", "--output-directory", help="Output directory", default=None)
    parser.add_argument("-n", "--n-samples-per-iteration", help="Number of samples to evaluate per iteration",
                        type=int, default=10)
    parser.add_argument("-s", "--seed", help="Random number generator seed", default=None, type=int)
    parser.add_argument("-f", "--objective", help="Objective type ['RKHSObjective', 'MixtureObjective']",
                        default='RKHSObjective')
    parser.add_argument("-p", "--show-plots", help="Show intermediate plots", action='store_true')
    cl_args = parser.parse_args()

    main(cl_args)
    print("Done")
