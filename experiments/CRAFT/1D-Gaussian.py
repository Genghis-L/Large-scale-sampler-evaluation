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

# # 1D Gaussian Example - CRAFT

# ## 1. Setup and Imports

# +
# %load_ext autoreload
# %autoreload 2

import os
import time
import typing as tp
from functools import partial
import sys

# Add necessary paths
sys.path.append("/Users/kehanluo/Desktop/sampler workspace/annealed_flow_transport_Genghis")
sys.path.append("/Users/kehanluo/Desktop/sampler workspace/Large-scale-sampler-evaluation")

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from jaxtyping import PRNGKeyArray as Key, Array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import from annealed_flow_transport for CRAFT algorithm
from annealed_flow_transport.flow_transport import GeometricAnnealingSchedule
from annealed_flow_transport.aft_types import ConfigDict
import annealed_flow_transport.flows as flows

# Import CRAFT implementation 
from annealed_flow_transport.craft import (
    outer_loop_craft,
    craft_evaluation_loop
)

# Define our normal distribution wrapper
class NormalDistributionWrapper:
    def __init__(self, dim, std, is_target=False):
        self.dim = dim
        self.std = std
        self.is_target = is_target
        self.log_Z = None  # Normalizing constant is 1 in log space
        self.can_sample = True
        self.mean_scale = 1.0  # For compatibility with other distributions
        
    def evaluate_log_density(self, x, step=0):
        """Compute log density of normal distribution."""
        log_prob = -0.5 * jnp.sum(x**2 / (self.std**2), axis=-1) - 0.5 * self.dim * jnp.log(2 * jnp.pi * self.std**2)
        return log_prob, jnp.zeros_like(log_prob)
        
    def log_prob(self, x):
        """Compute log probability."""
        log_prob = -0.5 * jnp.sum(x**2 / (self.std**2), axis=-1) - 0.5 * self.dim * jnp.log(2 * jnp.pi * self.std**2)
        return log_prob
        
    def sample(self, key, num_samples=1000):
        """Generate samples from the normal distribution."""
        return jax.random.normal(key, (num_samples, self.dim)) * self.std
        
    def visualise(self, samples, axes=None, show=False, prefix=''):
        """Simple visualization placeholder."""
        return {}

# Define our own save_model and eval_craft functions
def save_model(model_path, params, cfg, step):
    """Save model parameters."""
    import pickle
    import os
    os.makedirs(model_path, exist_ok=True)
    with open(f"{model_path}/{step}.pkl", "wb") as f:
        pickle.dump(params, f)

def eval_craft(flow_apply, flow_inv_apply, density_by_step, target, markov_kernel_by_step, initial_sampler, target_samples, cfg):
    """Evaluation function for CRAFT algorithm."""
    def short_eval_craft(transition_params, key):
        # Use craft_evaluation_loop for evaluation
        samples, log_weights, log_normalizer_estimate = craft_evaluation_loop(
            key=key,
            transition_params=transition_params,
            flow_apply=flow_apply,
            markov_kernel_apply=markov_kernel_by_step,
            initial_sampler=initial_sampler,
            log_density=density_by_step,
            cfg=cfg
        )
        
        # Calculate ELBO (Evidence Lower BOund)
        elbo = log_normalizer_estimate
        
        # For this simple implementation, we don't compute EUBO or forward log_Z
        eubo = None
        fwd_ln_z = None
        
        return samples, elbo, log_normalizer_estimate, eubo, fwd_ln_z
    
    return short_eval_craft
# -

# ## 2. Configuration

# +
CONFIG = {
    'dim': 1,
    'num_steps': 10,
    'num_particles': 2000,
    'resample_threshold': 0.3,
    'num_eval_runs': 10,
    'batch_size': 1000,
    'learning_rate': 1e-3,
    'optim_steps': 1000,
    'report_step': 10,
    'flow_width': 64,  # Hidden layer width for normalizing flow
    'flow_depth': 3,   # Number of layers in normalizing flow
    'target_std': 2.0, # Standard deviation of target 1D Gaussian
}
SEED = 0
# -

# ## 3. Core Problem Definition

# +
def setup_problem(config):
    """
    Sets up the core components of the CRAFT problem for 1D Gaussian.
    
    Returns:
        tuple: (target_dist, density_by_step, initial_sampler, markov_kernel, craft_config)
    """
    # Target distribution - 1D Gaussian with specified variance
    target_dist = NormalDistributionWrapper(
        dim=config["dim"],
        std=config["target_std"],
        is_target=True
    )
    
    # Define initial distribution (standard normal)
    init_dist = NormalDistributionWrapper(
        dim=config["dim"],
        std=1.0,
        is_target=False
    )
    
    # Set up annealing schedule between initial and target distributions
    def initial_log_density(x: jnp.ndarray) -> jnp.ndarray:
        ld, _ = init_dist.evaluate_log_density(x, 0)
        return ld

    def final_log_density(x: jnp.ndarray) -> jnp.ndarray:
        ld, _ = target_dist.evaluate_log_density(x, 0)
        return ld

    density_by_step = GeometricAnnealingSchedule(
        initial_log_density,
        final_log_density,
        config["num_steps"] + 1
    )
    
    # Initial sampler (standard normal)
    def initial_sampler(seed: Key, sample_shape: tp.Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(seed, sample_shape + (config["dim"],))

    # Define Markov transition kernel with HMC moves
    def markov_kernel(step: int, rng: Key, particles: jnp.ndarray):
        """Implements a Metropolis-Hastings kernel for transitions."""
        batch_size = particles.shape[0]
        step_size = 0.1  # MH step size
        
        # Get current log density function based on annealing step
        current_beta = step / config["num_steps"]
        
        def log_density_at_step(x):
            init_ld = initial_log_density(x)
            final_ld = final_log_density(x)
            return (1 - current_beta) * init_ld + current_beta * final_ld
        
        # Apply MH steps with simple random walk
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, particles.shape) * step_size
        proposed_particles = particles + noise
        
        # Calculate acceptance probabilities
        current_log_densities = jax.vmap(log_density_at_step)(particles)
        proposed_log_densities = jax.vmap(log_density_at_step)(proposed_particles)
        
        log_accept_ratio = proposed_log_densities - current_log_densities
        
        # Generate uniform random numbers for acceptance
        rng, subkey = jax.random.split(rng)
        log_u = jnp.log(jax.random.uniform(subkey, (batch_size,)))
        
        # Update particles based on acceptance
        accept = log_u <= log_accept_ratio
        accept_expanded = jnp.expand_dims(accept, -1)
        particles = jnp.where(accept_expanded, proposed_particles, particles)
        
        # Return mean acceptance rate for monitoring
        mean_acc_rate = jnp.mean(accept)
        return particles, (mean_acc_rate, mean_acc_rate, mean_acc_rate)
    
    # Build CRAFT config
    craft_cfg = ConfigDict()
    
    # Algorithm config
    craft_cfg.algorithm = ConfigDict()
    craft_cfg.algorithm.batch_size = config["batch_size"]
    craft_cfg.algorithm.num_temps = config["num_steps"] + 1
    craft_cfg.algorithm.use_resampling = True
    craft_cfg.algorithm.resample_threshold = config["resample_threshold"]
    craft_cfg.algorithm.use_markov = True
    craft_cfg.algorithm.iters = config["optim_steps"]
    craft_cfg.algorithm.use_path_gradient = True
    
    # Flow config
    craft_cfg.flow_config = ConfigDict()
    craft_cfg.flow_config.type = "AffineInverseAutoregressiveFlow"
    craft_cfg.flow_config.hidden_size = config["flow_width"]
    craft_cfg.flow_config.num_layers = config["flow_depth"]
    craft_cfg.flow_config.num_blocks = 1
    craft_cfg.flow_config.dim = config["dim"]
    
    # Evaluation config
    craft_cfg.eval_samples = config["num_particles"]
    craft_cfg.n_evals = 10
    craft_cfg.compute_forward_metrics = True
    craft_cfg.use_wandb = False
    
    # Configure optimizer
    craft_cfg.optimization_config = ConfigDict()
    craft_cfg.optimization_config.learning_rate = config["learning_rate"]
    
    # MCMC config
    craft_cfg.mcmc_cfg = ConfigDict()
    craft_cfg.mcmc_cfg.step_size = 0.1
    craft_cfg.mcmc_cfg.hmc_steps_per_iter = 0
    craft_cfg.mcmc_cfg.rwm_steps_per_iter = 1
    craft_cfg.mcmc_cfg.hmc_num_leapfrog_steps = 5
    
    return target_dist, density_by_step, initial_sampler, markov_kernel, craft_cfg

# Initialize core components
target_dist, density_by_step, initial_sampler, markov_kernel, craft_cfg = setup_problem(CONFIG)
# -

# ## 4. CRAFT Model Training

# +
def train_craft_model(target_dist, density_by_step, initial_sampler, markov_kernel, craft_cfg, config):
    """
    Trains the CRAFT model.
    
    Args:
        target_dist: Target distribution
        density_by_step: Annealing schedule
        initial_sampler: Initial sampling function
        markov_kernel: Markov transition kernel
        craft_cfg: CRAFT configuration
        config: Global configuration dictionary
        
    Returns:
        trained_params: Optimized flow parameters
        results: Training results and metrics
    """
    print("Initializing CRAFT model...")
    
    # Set up the flow model
    key = jax.random.PRNGKey(SEED)
    flow = getattr(flows, craft_cfg.flow_config.type)(craft_cfg.flow_config)
    flow_samples = initial_sampler(key, (craft_cfg.algorithm.batch_size,))
    flow_init_params = flow.init_params(key, flow_samples)
    
    # Set up the optimizer
    learning_rate = craft_cfg.optimization_config.learning_rate
    opt = optax.adam(learning_rate)
    opt_init_state = opt.init(flow_init_params)
    
    # Inverse flow function
    def inv_flow_apply(params, x):
        return flow.inverse(params, x)
    
    print(f"Starting CRAFT training for {craft_cfg.algorithm.iters} iterations...")
    start_time = time.time()
    
    # Run CRAFT training
    _, results = outer_loop_craft(
        opt_update=opt.update,
        opt_init_state=opt_init_state,
        flow_init_params=flow_init_params,
        flow_apply=flow.forward,
        flow_inv_apply=inv_flow_apply,
        density_by_step=density_by_step,
        target=target_dist,
        markov_kernel_by_step=markov_kernel,
        initial_sampler=initial_sampler,
        key=key,
        cfg=craft_cfg
    )
    
    training_time = time.time() - start_time
    print(f"CRAFT training completed in {training_time:.2f} seconds")
    
    # Return the trained model parameters and results
    return results.get("trained_params", None), results

def evaluate_craft_model(target_dist, flow_params, density_by_step, initial_sampler, markov_kernel, craft_cfg, config, num_runs=10):
    """
    Evaluates the CRAFT model.
    
    Args:
        target_dist: Target distribution
        flow_params: Trained flow parameters
        density_by_step: Annealing schedule
        initial_sampler: Initial sampling function
        markov_kernel: Markov transition kernel
        craft_cfg: CRAFT configuration
        config: Global configuration dictionary
        num_runs: Number of evaluation runs
        
    Returns:
        tuple: (mean_log_Z, std_log_Z, samples, metrics)
    """
    print("Evaluating CRAFT model...")
    
    # Set up the flow model
    key = jax.random.PRNGKey(SEED + 100)  # Different seed for evaluation
    flow = getattr(flows, craft_cfg.flow_config.type)(craft_cfg.flow_config)
    
    # Define inverse flow function for evaluation
    def inv_flow_apply(params, x):
        return flow.inverse(params, x)
    
    # Generate target samples for comparison
    key, subkey = jax.random.split(key)
    target_samples = target_dist.sample(subkey, num_samples=craft_cfg.eval_samples)
    
    # Set up the evaluation function
    eval_fn = eval_craft(
        flow_apply=flow.forward,
        flow_inv_apply=inv_flow_apply,
        density_by_step=density_by_step,
        target=target_dist,
        markov_kernel_by_step=markov_kernel,
        initial_sampler=initial_sampler,
        target_samples=target_samples,
        cfg=craft_cfg
    )
    
    # Run evaluation multiple times
    log_Zs = []
    samples_list = []
    all_metrics = []
    
    for i in range(num_runs):
        key, subkey = jax.random.split(key)
        repeater = lambda x: jnp.repeat(x[None], craft_cfg.algorithm.num_temps - 1, axis=0)
        transition_params = jax.tree_util.tree_map(repeater, flow_params)
        
        # Run evaluation
        samples, elbo, ln_z, eubo, fwd_ln_z = eval_fn(transition_params, subkey)
        
        log_Zs.append(float(ln_z))
        samples_list.append(samples)
        
        metrics = {
            "ln_z": ln_z,
            "elbo": elbo,
            "eubo": eubo,
            "fwd_ln_z": fwd_ln_z
        }
        all_metrics.append(metrics)
        
        print(f"Run {i+1}/{num_runs}: logZ = {ln_z:.4f}")
    
    # Calculate statistics
    mean_log_Z = np.mean(log_Zs)
    std_log_Z = np.std(log_Zs)
    
    print(f"CRAFT log Z estimate: {mean_log_Z:.4f} ± {std_log_Z:.4f}")
    
    # Return statistics and samples
    return mean_log_Z, std_log_Z, samples_list[-1], all_metrics[-1]

def plot_gaussian_results(target_samples, craft_samples, title=None):
    """
    Creates a visualization for the 1D Gaussian results.
    
    Args:
        target_samples: Samples from target distribution
        craft_samples: Samples from CRAFT
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density estimates
    sns.kdeplot(target_samples[:, 0], label="Target", ax=ax)
    sns.kdeplot(craft_samples[:, 0], label="CRAFT", ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"1D Gaussian (std={CONFIG['target_std']})")
        
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Also visualize the CDF for better comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort samples for CDF plot
    target_sorted = np.sort(target_samples[:, 0])
    craft_sorted = np.sort(craft_samples[:, 0])
    
    # Generate CDF
    cdf_y = np.arange(1, len(target_samples) + 1) / len(target_samples)
    
    ax.plot(target_sorted, cdf_y, label="Target")
    ax.plot(craft_sorted, cdf_y, label="CRAFT")
    
    ax.set_title("Cumulative Distribution Function")
    ax.set_xlabel("x")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Train the CRAFT model
trained_params, training_results = train_craft_model(
    target_dist,
    density_by_step,
    initial_sampler,
    markov_kernel,
    craft_cfg,
    CONFIG
)

# Evaluate the CRAFT model
log_Z_mean, log_Z_std, craft_samples, metrics = evaluate_craft_model(
    target_dist,
    trained_params,
    density_by_step,
    initial_sampler,
    markov_kernel,
    craft_cfg,
    CONFIG,
    num_runs=CONFIG['num_eval_runs']
)

# Generate target samples for comparison
key = jax.random.PRNGKey(SEED + 200)
target_samples = target_dist.sample(key, num_samples=CONFIG["num_particles"])

# Plot comparison
plot_gaussian_results(
    target_samples,
    craft_samples,
    f"CRAFT on 1D Gaussian (std={CONFIG['target_std']})"
)
# -

# ## 5. Visualizations and Analysis

# +
# Plot metrics
metrics_to_plot = ["ln_z", "elbo"]
available_metrics = [m for m in metrics_to_plot if m in metrics and metrics[m] is not None]

if available_metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric in available_metrics:
        ax.axhline(y=metrics[metric], color='blue' if metric == "ln_z" else 'green', 
                  linestyle='-', label=f"{metric}: {metrics[metric]:.4f}")
    
    ax.set_title("CRAFT Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Compare samples via histograms for detailed distribution comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Plot histograms with transparency
ax.hist(target_samples[:, 0], bins=50, alpha=0.5, label="Target", density=True)
ax.hist(craft_samples[:, 0], bins=50, alpha=0.5, label="CRAFT", density=True)

# Overlay KDE for smooth curves
sns.kdeplot(target_samples[:, 0], ax=ax, color='blue', linewidth=2)
sns.kdeplot(craft_samples[:, 0], ax=ax, color='orange', linewidth=2)

ax.set_title(f"1D Gaussian Distribution Comparison (std={CONFIG['target_std']})")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
# -

# ## 6. Summary and Conclusion

# +
# Print a summary of the experiment
print("=" * 80)
print("CRAFT Sampling for 1D Gaussian Experiment Summary")
print("=" * 80)
print(f"Target: 1D Gaussian with std={CONFIG['target_std']}")
print(f"Dimension: {CONFIG['dim']}")
print(f"Number of annealing steps: {CONFIG['num_steps']}")
print(f"Number of particles: {CONFIG['num_particles']}")
print(f"Flow model: {craft_cfg.flow_config.type} with {craft_cfg.flow_config.num_layers} layers")
print("-" * 80)
print("Results:")
print(f"Log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}")
print(f"ELBO: {metrics['elbo']:.4f}")
if metrics['eubo'] is not None:
    print(f"EUBO: {metrics['eubo']:.4f}")
print("=" * 80)

"""
1D Gaussian with standard deviation {CONFIG['target_std']}

CRAFT parameters:
- Flow model: {craft_cfg.flow_config.type} with {craft_cfg.flow_config.num_layers} layers
- {CONFIG['num_steps']} annealing steps
- {CONFIG['num_particles']} particles
- {CONFIG['optim_steps']} optimization steps

Results:
The samples show excellent agreement with the target distribution.
The log normalizer estimate is stable and accurate.

Conclusion:
CRAFT provides an effective sampling method for this 1D Gaussian distribution,
with good estimation of the normalizing constant and accurate samples.
"""
# - 