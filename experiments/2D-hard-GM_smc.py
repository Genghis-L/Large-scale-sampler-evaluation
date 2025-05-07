# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: '1.16.7'
#   kernelspec:
#     display_name: pdds
#     language: python
#     name: python3
# ---

# # Hard 2D Gaussian Mixture - SMC

# ## 1. Setup and Imports
# +
# %load_ext autoreload
# %autoreload 2

import os
import time
import typing as tp
from functools import partial

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray as Key, Array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from annealed_flow_transport.smc import fast_outer_loop_smc, outer_loop_smc
from annealed_flow_transport.flow_transport import GeometricAnnealingSchedule
from annealed_flow_transport.aft_types import ConfigDict

from pdds.distributions import ChallengingTwoDimensionalMixture
# -

# ## 2. Configuration
# +
CONFIG = {
    'dim': 2,
    'mean_scale': 1.0,
    'num_steps': 10,
    'num_particles': 2000,
    'resample_threshold': 0.3,
    'num_eval_runs': 10,
    'report_step': 1,
    'mcmc_steps_per_iter': 5,  # Number of MCMC steps per iteration
    'step_size': 0.1           # Step size for MCMC transition
}
SEED = 0
# -

# ## 3. Core Problem Definition
# +
def setup_problem(config):
    """
    Sets up the core components of the SMC problem for harder Gaussian mixture.
    
    Returns:
        tuple: (target_dist, density_by_step, initial_sampler, markov_kernel, smc_config)
    """
    # Target distribution - More challenging 2D Gaussian mixture with larger separation
    target_dist = ChallengingTwoDimensionalMixture(
        mean_scale=config["mean_scale"], 
        dim=config["dim"], 
        is_target=True
    )
    
    # Geometric annealing schedule
    def initial_log_density(x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * config["dim"] * jnp.log(2 * jnp.pi)

    def final_log_density(x: jnp.ndarray) -> jnp.ndarray:
        ld, _ = target_dist.evaluate_log_density(x, 0)
        return ld

    density_by_step = GeometricAnnealingSchedule(
        initial_log_density,
        final_log_density,
        config["num_steps"] + 1
    )
    
    # Initial sampler (standard normal) - Forward kernel at time 0
    def initial_sampler(rng: Key, batch_size: int, sample_shape: tp.Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(rng, (batch_size,) + sample_shape)
    
    # MCMC transition kernel (proper Markov kernel with HMC moves)
    def markov_kernel(step: int, rng: Key, particles: jnp.ndarray):
        """
        Implements a Metropolis-Hastings MCMC kernel for transitions.
        This acts as the forward kernel in SMC between consecutive distributions.
        """
        batch_size = particles.shape[0]
        dim = particles.shape[1]
        step_size = config["step_size"]
        mcmc_steps = config["mcmc_steps_per_iter"]
        
        # Compute current annealing parameter
        current_beta = step / config["num_steps"]

        # Loop over multiple MCMC steps, computing densities in batch
        acc_rates = []
        for mcmc_step in range(mcmc_steps):
            # Split the RNG key for this step
            rng, subkey = jax.random.split(rng)

            # Propose moves with random walk
            noise = jax.random.normal(subkey, particles.shape) * step_size
            proposed_particles = particles + noise

            # Compute log densities for current and proposed batches
            log_init_particles = initial_log_density(particles)
            log_init_proposed = initial_log_density(proposed_particles)
            log_final_particles = final_log_density(particles)
            log_final_proposed = final_log_density(proposed_particles)
            current_log_densities = (1 - current_beta) * log_init_particles + current_beta * log_final_particles
            proposed_log_densities = (1 - current_beta) * log_init_proposed + current_beta * log_final_proposed

            # Compute acceptance probabilities
            log_accept_ratio = proposed_log_densities - current_log_densities
            
            # Generate uniform random numbers for acceptance
            rng, subkey = jax.random.split(rng)
            log_u = jnp.log(jax.random.uniform(subkey, (batch_size,)))
            
            # Update particles based on acceptance
            accept = log_u <= log_accept_ratio
            accept_expanded = jnp.expand_dims(accept, -1)
            particles = jnp.where(accept_expanded, proposed_particles, particles)
            
            # Track acceptance rate
            acc_rates.append(jnp.mean(accept))
        
        # Return the moved particles and acceptance rates
        mean_acc_rate = jnp.mean(jnp.array(acc_rates))
        return particles, (mean_acc_rate, mean_acc_rate, mean_acc_rate)
    
    # Build SMC config
    smc_cfg = ConfigDict()
    smc_cfg.batch_size = config["num_particles"]
    smc_cfg.sample_shape = (config["dim"],)
    smc_cfg.num_temps = config["num_steps"] + 1
    smc_cfg.use_resampling = True
    smc_cfg.resample_threshold = config["resample_threshold"]
    smc_cfg.report_step = config["report_step"]
    
    return target_dist, density_by_step, initial_sampler, markov_kernel, smc_cfg

# Initialize core components
target_dist, density_by_step, initial_sampler, markov_kernel, smc_cfg = setup_problem(CONFIG)
# -

# ## 4. SMC Evaluation
# +
def evaluate_sampler(target_dist, density_by_step, initial_sampler, markov_kernel, smc_cfg, num_eval_runs=10):
    """
    Evaluates the SMC sampler by running it multiple times and calculating statistics.
    
    Args:
        target_dist: Target distribution
        density_by_step: Annealing schedule
        initial_sampler: Initial sampling function (forward kernel at time 0)
        markov_kernel: Markov transition kernel (forward kernel for transitions)
        smc_cfg: SMC configuration
        num_eval_runs: Number of evaluation runs
        
    Returns:
        tuple: (mean_log_Z, std_log_Z, final_samples, all_log_Zs, elapsed_times)
    """
    # Create the evaluation sampler; bind all args except RNG key
    sampler = jax.jit(lambda rng_key: fast_outer_loop_smc(
        density_by_step,
        initial_sampler,
        markov_kernel_by_step=markov_kernel,
        key=rng_key,
        config=smc_cfg,
    ))
    
    # Evaluate the normalizing constant estimate over multiple runs
    log_Zs = np.zeros(num_eval_runs)
    elapsed_times = np.zeros(num_eval_runs)
    key_eval = jax.random.PRNGKey(SEED + 100)  # Different seed for evaluation
    particle_state = None
    
    for i in range(num_eval_runs):
        key_eval, subkey = jax.random.split(key_eval)
        start_time = time.time()
        particle_state = sampler(subkey)
        elapsed_times[i] = time.time() - start_time
        log_Zs[i] = float(particle_state.log_normalizer_estimate)
        print(f"Run {i+1}/{num_eval_runs}: logZ = {log_Zs[i]:.4f}, time = {elapsed_times[i]:.2f}s")
    
    # Resample the final particles from the last run
    samples = particle_state.samples
    log_weights = particle_state.log_weights
    
    # Resample final particles
    norm_logw = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(norm_logw)
    key_eval, subkey = jax.random.split(key_eval)
    indices = jax.random.choice(
        subkey, 
        a=CONFIG["num_particles"], 
        shape=(CONFIG["num_particles"],), 
        replace=True, 
        p=weights
    )
    resampled = samples[indices]
    
    return np.mean(log_Zs), np.std(log_Zs), resampled, log_Zs, elapsed_times

def plot_gaussian_mixture_results(target_samples, samples, label, title=None):
    """
    Creates a simplified visualization for Gaussian mixture results with 
    2D scatter plot and 1D marginals in a single row.
    
    Args:
        target_samples: Samples from target distribution
        samples: Samples from approximation method
        label: Label for the approximation method
        title: Optional title
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 2D scatter
    axs[0].scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
    axs[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, label=label, s=10)
    
    if title:
        axs[0].set_title(title)
    else:
        axs[0].set_title(f"Hard 2D Gaussian Mixture (mean_scale={CONFIG['mean_scale']})")
        
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')
    
    # Plot 1D marginal x-axis
    sns.kdeplot(target_samples[:, 0], label="Target", ax=axs[1])
    sns.kdeplot(samples[:, 0], label=label, ax=axs[1])
    axs[1].set_title("1D Marginal (x-axis)")
    axs[1].set_xlabel("x")
    axs[1].legend()
    
    # Plot 1D marginal y-axis
    sns.kdeplot(target_samples[:, 1], label="Target", ax=axs[2])
    sns.kdeplot(samples[:, 1], label=label, ax=axs[2])
    axs[2].set_title("1D Marginal (y-axis)")
    axs[2].set_xlabel("y")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
# -

# ## 5. Run SMC Experiments over mean scales 1 to 5
mean_scales = range(1, 6)
results_summary = []
for ms in mean_scales:
    CONFIG['mean_scale'] = ms
    print(f"\n=== Running SMC for mean_scale={ms} ===")
    # Re-initialize core components for this mean_scale
    target_dist, density_by_step, initial_sampler, markov_kernel, smc_cfg = setup_problem(CONFIG)
    # Generate target samples for comparison
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key)
    target_samples = target_dist.sample(subkey, num_samples=CONFIG['num_particles'])
    # Evaluate sampler
    log_Z_mean, log_Z_std, smc_samples, all_log_Zs, elapsed_times = evaluate_sampler(
        target_dist, density_by_step, initial_sampler, markov_kernel, smc_cfg, CONFIG['num_eval_runs']
    )
    print(f"SMC log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}")
    print(f"Average runtime: {np.mean(elapsed_times):.2f}s ± {np.std(elapsed_times):.2f}s")
    # Plot comparison
    plot_gaussian_mixture_results(
        target_samples,
        smc_samples,
        f"SMC (mean_scale={ms})",
        f"SMC on Hard 2D Gaussian Mixture (mean_scale={ms})"
    )
    results_summary.append((ms, log_Z_mean, log_Z_std, np.mean(elapsed_times), np.std(elapsed_times)))
# Consolidated summary across all mean_scale values
print("\n=== Summary over mean_scales ===")
for ms, m_val, s_val, t_mean, t_std in results_summary:
    print(f"mean_scale={ms}: logZ={m_val:.4f}±{s_val:.4f}, time={t_mean:.2f}s±{t_std:.2f}s")

# ## 6. Additional Visualizations and Analysis
# +
# # Plot logZ distribution
# plt.figure(figsize=(10, 6))
# plt.hist(all_log_Zs, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
# plt.axvline(log_Z_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {log_Z_mean:.4f}')
# plt.axvline(log_Z_mean - log_Z_std, color='green', linestyle=':', linewidth=2, 
#             label=f'Std Dev: {log_Z_std:.4f}')
# plt.axvline(log_Z_mean + log_Z_std, color='green', linestyle=':', linewidth=2)
# plt.title('Distribution of Log Normalizing Constant Estimates')
# plt.xlabel('Log Z')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# # Plot runtime distribution
# plt.figure(figsize=(10, 6))
# plt.hist(elapsed_times, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
# plt.axvline(np.mean(elapsed_times), color='red', linestyle='--', linewidth=2, 
#             label=f'Mean: {np.mean(elapsed_times):.2f}s')
# plt.title('Distribution of Runtime')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

# Plot 2D density comparison
plt.figure(figsize=(12, 5))

# Target density
plt.subplot(1, 2, 1)
x, y = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
pos = np.dstack((x, y))
pos_flat = pos.reshape(-1, 2)

# Compute target density (approximated via KDE from samples)
from scipy.stats import gaussian_kde
target_kde = gaussian_kde(target_samples.T)
target_density = target_kde(pos_flat.T).reshape(100, 100)

plt.contourf(x, y, target_density, levels=50, cmap='viridis')
plt.title('Target Density')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# SMC sampled density
plt.subplot(1, 2, 2)
smc_kde = gaussian_kde(np.asarray(smc_samples).T)
smc_density = smc_kde(pos_flat.T).reshape(100, 100)

plt.contourf(x, y, smc_density, levels=50, cmap='viridis')
plt.title('SMC Density')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.tight_layout()
plt.show()
# -

# ## 7. Summary and Conclusion
# +
# Print a summary of the experiment
print("=" * 80)
print("SMC Sampling for Hard 2D Gaussian Mixture Experiment Summary")
print("=" * 80)
print(f"Target: Hard 2D Gaussian Mixture with mean_scale={CONFIG['mean_scale']}")
print(f"Dimension: {CONFIG['dim']}")
print(f"Number of annealing steps: {CONFIG['num_steps']}")
print(f"Number of particles: {CONFIG['num_particles']}")
print(f"Resampling threshold: {CONFIG['resample_threshold']}")
print(f"MCMC steps per iteration: {CONFIG['mcmc_steps_per_iter']}")
print(f"MCMC step size: {CONFIG['step_size']}")
print("-" * 80)
print("Results:")
print(f"Log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}")
print(f"Average runtime: {np.mean(elapsed_times):.2f}s ± {np.std(elapsed_times):.2f}s")
print("=" * 80)

"""
Hard 2D Gaussian Mixture with mean_scale=3

SMC parameters:
- Geometric annealing
- Metropolis-Hastings MCMC transition kernel
- 10 annealing steps
- 2000 particles
- 5 MCMC steps per iteration

Results:
SMC works well for mean_scale is small, but fails for large mean_scale.
"""
# - 