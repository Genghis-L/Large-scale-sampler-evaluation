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

# # Hard 2D Gaussian Mixture - AFT SMC

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
    'mean_scale': 3.0,
    'num_steps': 10,
    'num_particles': 2000,
    'resample_threshold': 0.3,
    'num_eval_runs': 10,
    'report_step': 1
}
SEED = 0
# -

# ## 3. Core Problem Definition
# +
def setup_problem(config):
    """
    Sets up the core components of the AFT SMC problem for harder Gaussian mixture.
    
    Returns:
        tuple: (target_dist, density_by_step, initial_sampler, markov_kernel, aft_config)
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
    
    # Initial sampler (standard normal)
    def initial_sampler(rng: Key, batch_size: int, sample_shape: tp.Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(rng, (batch_size,) + sample_shape)

    # Identity Markov kernel (no MCMC moves)
    def markov_kernel(step: int, rng: Key, particles: jnp.ndarray):
        acc = (jnp.ones_like(particles[:, 0]), jnp.ones_like(particles[:, 0]), jnp.ones_like(particles[:, 0]))
        return particles, acc
    
    # Build AFT config
    aft_cfg = ConfigDict()
    aft_cfg.batch_size = config["num_particles"]
    aft_cfg.sample_shape = (config["dim"],)
    aft_cfg.num_temps = config["num_steps"] + 1
    aft_cfg.use_resampling = True
    aft_cfg.resample_threshold = config["resample_threshold"]
    aft_cfg.report_step = config["report_step"]
    
    return target_dist, density_by_step, initial_sampler, markov_kernel, aft_cfg

# Initialize core components
target_dist, density_by_step, initial_sampler, markov_kernel, aft_cfg = setup_problem(CONFIG)
# -

# ## 4. AFT SMC Evaluation
# +
def evaluate_sampler(target_dist, density_by_step, initial_sampler, markov_kernel, aft_cfg, num_eval_runs=10):
    """
    Evaluates the AFT SMC sampler by running it multiple times and calculating statistics.
    
    Args:
        target_dist: Target distribution
        density_by_step: Annealing schedule
        initial_sampler: Initial sampling function
        markov_kernel: Markov transition kernel
        aft_cfg: AFT SMC configuration
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
        config=aft_cfg,
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

# ## 5. Run AFT SMC Experiment
# +
# Generate target samples for comparison
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)
target_samples = target_dist.sample(subkey, num_samples=CONFIG["num_particles"])

# Run fast AFT SMC multiple times to get statistics
print("Evaluating AFT SMC sampler...")
log_Z_mean, log_Z_std, aft_samples, all_log_Zs, elapsed_times = evaluate_sampler(
    target_dist, 
    density_by_step, 
    initial_sampler, 
    markov_kernel, 
    aft_cfg, 
    CONFIG["num_eval_runs"]
)

print(f'AFT SMC log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}')
print(f'Average runtime: {np.mean(elapsed_times):.2f}s ± {np.std(elapsed_times):.2f}s')

# Plot comparison
plot_gaussian_mixture_results(
    target_samples, 
    aft_samples, 
    "AFT SMC", 
    f"AFT SMC on Hard 2D Gaussian Mixture (mean_scale={CONFIG['mean_scale']})"
)
# -

# ## 6. Additional Visualizations and Analysis
# +
# Plot logZ distribution
plt.figure(figsize=(10, 6))
plt.hist(all_log_Zs, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(log_Z_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {log_Z_mean:.4f}')
plt.axvline(log_Z_mean - log_Z_std, color='green', linestyle=':', linewidth=2, 
            label=f'Std Dev: {log_Z_std:.4f}')
plt.axvline(log_Z_mean + log_Z_std, color='green', linestyle=':', linewidth=2)
plt.title('Distribution of Log Normalizing Constant Estimates')
plt.xlabel('Log Z')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot runtime distribution
plt.figure(figsize=(10, 6))
plt.hist(elapsed_times, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(np.mean(elapsed_times), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(elapsed_times):.2f}s')
plt.title('Distribution of Runtime')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

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

# AFT SMC sampled density
plt.subplot(1, 2, 2)
aft_kde = gaussian_kde(np.asarray(aft_samples).T)
aft_density = aft_kde(pos_flat.T).reshape(100, 100)

plt.contourf(x, y, aft_density, levels=50, cmap='viridis')
plt.title('AFT SMC Density')
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
print("AFT SMC Sampling for Hard 2D Gaussian Mixture Experiment Summary")
print("=" * 80)
print(f"Target: Hard 2D Gaussian Mixture with mean_scale={CONFIG['mean_scale']}")
print(f"Dimension: {CONFIG['dim']}")
print(f"Number of annealing steps: {CONFIG['num_steps']}")
print(f"Number of particles: {CONFIG['num_particles']}")
print(f"Resampling threshold: {CONFIG['resample_threshold']}")
print("-" * 80)
print("Results:")
print(f"Log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}")
print(f"Average runtime: {np.mean(elapsed_times):.2f}s ± {np.std(elapsed_times):.2f}s")
print("=" * 80)

"""
Hard 2D Gaussian Mixture with mean_scale=3

AFT SMC parameters:
- Geometric annealing
- No MCMC moves
- 10 annealing steps
- 2000 particles

Results:
The samples show good coverage of the target distribution's modes.
The log normalizer estimate is stable across multiple runs.

Conclusion:
AFT SMC provides an effective sampling method for this challenging
multimodal distribution even without additional MCMC refinement.
"""
# - 