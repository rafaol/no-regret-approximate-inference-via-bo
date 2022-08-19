import argparse
import os
import random
import time
import yaml

import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal

from dbo.af import PlainUCB
from dbo.gp import LFIGPModel
from dbo.sampling import MCMCSampler, density_estimator, ScipyKDE
from dbo.util import cd, make_grid, save_object, plot_estimates, configure_matplotlib, load_abc_samples
from dbo.diagnostics import sample_divergence, gs_divergence
from dbo.lfi import RLObjective


def run_experiment(dimensionality, n_it, n_samples_per_it, n_chains, noise_sd,
                   output_directory, n_test=2000, n_burn=400, show_plots=False):
    problem_prior = MultivariateNormal(torch.zeros(dimensionality), torch.eye(dimensionality))
    objective = RLObjective(n_dim=dimensionality)

    gp_model = LFIGPModel(n_dim=dimensionality, prior=problem_prior)
    gp_model.covar_module.base_kernel.lengthscale = torch.tensor([.75, .75])
    gp_model.covar_module.outputscale = 40.

    f_norm = 3
    obs_min = -1000.

    objective_samples = load_abc_samples()
    true_posterior = ScipyKDE(objective_samples)

    gp_model.likelihood.noise = noise_sd**2
    acq_fun = PlainUCB(gp_model, f_bound=f_norm, sigma_out=0)
    af_sampler = MCMCSampler(acq_fun, problem_prior, n_chains=n_chains, use_jit=True, sampler='EMCEE')

    n_points = 100
    x_lb = problem_prior.mean.mean().item() - 3 * problem_prior.variance.mean().sqrt().item()
    x_ub = problem_prior.mean.mean().item() + 3 * problem_prior.variance.mean().sqrt().item()

    r_test = None
    x_test = None
    if dimensionality == 2:
        x_test = make_grid(x_lb, x_ub, (x_ub - x_lb) / n_points)
        r_test = true_posterior.log_prob(x_test).cpu()

    divergences = torch.zeros(n_it)
    gs_divergences = torch.zeros(n_it)
    regret_bound = torch.zeros(n_it)
    n_kl_samples = 400

    # BO loop
    for t in range(n_it):
        print(f"Sampling... {t + 1}/{n_it}")
        x_t = x_test[acq_fun(x_test).argmax()].view(1, -1) + torch.randn(dimensionality)*0.05
        y_t = objective(x_t).view(1)

        # Debug
        print("Computing divergence...")
        test_samples = af_sampler.sample(n_kl_samples, n_burn=n_burn)
        gs_divergences[t] = gs_divergence(objective_samples, test_samples)
        divergences[t] = sample_divergence(objective_samples, test_samples)
        regret_bound[t] = 2 * acq_fun.beta_t * gp_model(test_samples).variance.sqrt().mean()
        print(f"Divergence/bound: {divergences[t]} < {regret_bound[t]}")
        print(f"gsKL divergence/bound: {gs_divergences[t]}/{regret_bound[t]}")

        if dimensionality == 2 and show_plots:
            plt.figure(figsize=(4, 8))
            af_test = af_sampler.un_dist.log_prob(x_test).cpu()
            v_min = -100
            v_max = 50
            plt.subplot(211)
            plot_estimates(r_test, x_lb, x_ub, vmin=v_min, vmax=v_max)
            plt.subplot(212)
            plot_estimates(af_test, x_lb, x_ub, vmin=v_min, vmax=v_max)
            plt.plot(test_samples[:, 0], test_samples[:, 1], 'm.', label="MCMC")
            plt.plot(x_t[:, 0], x_t[:, 1], 'ko', label="Evaluation points")
            plt.legend()
            plt.axis([x_lb, x_ub, x_lb, x_ub])
            plt.show()

        # Update models
        gp_model.update(x_t, y_t.clamp(min=obs_min))
        acq_fun.update()

    gp_sampler = MCMCSampler(gp_model.mean, problem_prior, n_chains=n_chains, use_jit=True, sampler='EMCEE')
    final_samples = gp_sampler.sample(n_test, n_burn=n_burn)
    final_de = density_estimator(final_samples)

    if dimensionality == 2 and show_plots:
        m_test = final_de.log_prob(x_test).exp().cpu()
        v_max = max(m_test.max().item(), r_test.exp().max().item())
        plt.figure(figsize=(4, 7))
        plt.subplot(211)
        plot_estimates(r_test.exp(), x_lb, x_ub, vmin=0, vmax=v_max)
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
    plt.xlabel("Evaluations")
    plt.legend()
    plt.tight_layout()
    with cd(output_directory):
        plt.savefig("result.png", dpi=300)
        torch.save(regret_bound, "bolfi-bounds.pth")
        torch.save(divergences, "bolfi-divergences.pth")
        torch.save(gs_divergences, "bolfi-gs_divergences.pth")
        torch.save(final_samples, "bolfi-final_samples.pth")
        save_object("bolfi-gp.pkl", gp_model)
        save_object("bolfi-objective.pkl", objective)

    if show_plots:
        plt.show()


def main(args):
    if args.output_directory is None:
        args.output_directory = os.path.join("experiments", time.strftime("bolfi-%Y-%m-%d-%H%M%S"))
    print(f"Writing files to {args.output_directory}")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    output_directory = args.output_directory

    if args.seed is None:
        seed = random.SystemRandom().getrandbits(32)
    else:
        seed = args.seed
    torch.manual_seed(seed)
    with cd(output_directory):
        with open("bolfi-seed.dat", "w") as f:
            f.write("{}\n".format(seed))
        with open("bolfi-args.yaml", "w") as f:
            yaml.dump(args, f)

    dim = 2
    log_lik_noise_sd = 4     # TODO: Estimate this from simulations
    n_mcmc_chains = 1
    n_samples_per_iteration = args.n_samples_per_iteration
    assert n_samples_per_iteration == 1
    n_iterations = args.n_iterations
    n_repeats = args.n_repeats

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
        run_experiment(dim, n_iterations, n_samples_per_iteration, n_mcmc_chains, log_lik_noise_sd,
                       run_dir, show_plots=args.show_plots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Likelihood-free inference with GP-UCB")
    parser.add_argument("-r", "--n-repeats", help="Number of repetitions", type=int, default=1)
    parser.add_argument("-t", "--n-iterations", help="Number of iterations", type=int, default=10)
    parser.add_argument("-o", "--output-directory", help="Output directory", default=None)
    parser.add_argument("-n", "--n-samples-per-iteration", help="Number of samples to evaluate per iteration",
                        type=int, default=1)
    parser.add_argument("-p", "--show-plots", help="Show intermediate plots", action='store_true')
    parser.add_argument("-s", "--seed", help="Random number generator seed", default=None, type=int)
    cl_args = parser.parse_args()

    main(cl_args)
    print("Done")
