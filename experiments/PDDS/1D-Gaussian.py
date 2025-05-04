# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: pdds
#     language: python
#     name: python3
# ---

# # Gaussian Interactive Example: Evaluating PDDS (PDDS)

# ## 1. Setup and Imports

# +
# %load_ext autoreload
# %autoreload 2

import os
import time
import warnings
import tqdm
import typing as tp
from functools import partial

# Set available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Filter Equinox warnings
warnings.filterwarnings("ignore", message="As of Equinox.*", category=UserWarning)

# JAX imports
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jaxtyping import PRNGKeyArray as Key, Array
from check_shapes import check_shapes

# Data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PDDS modules
from pdds.sde import SDE, guidance_loss, dsm_loss, LinearScheduler
from pdds.smc_problem import SMCProblem
from pdds.potentials import (
    RatioPotential, 
    NaivelyApproximatedPotential, 
    NNApproximatedPotential
)
from pdds.utils.shaping import broadcast
from pdds.utils.jax import _get_key_iter, x_gradient_stateful_parametrised
from pdds.utils.lr_schedules import loop_schedule
from pdds.ml_tools.state import TrainingState
from pdds.smc_loops import fast_outer_loop_smc
from pdds.distributions import NormalDistributionWrapper
from pdds.nn_models.mlp import PISGRADNet
from pdds.resampling import resampler
# -

# ## 2. Configuration

# +
# Global parameters
CONFIG = {
    # Problem dimensions
    "dim": 2,
    "sigma": 1.0,
    "t_0": 0.0,
    "t_f": 1.0,
    
    # Algorithm parameters
    "num_steps": 8,
    "num_particles": 10000,
    "batch_size": 300,
    
    # Target distribution parameters
    "target_mean": 2.75,
    "target_scale": 0.25,
    
    # SDE parameters
    "beta_0": 0.001,
    "beta_f": 12.0,
    
    # Training parameters
    "refresh_batch_every": 100,
    "optim_steps": 10000,
    "learning_rate": 1e-3,
    "lr_decay_rate": 0.95,
    "lr_transition_steps": 50,
    "ema_decay": 0.999,
}

# Initialize random key
SEED = 0
key = jax.random.PRNGKey(seed=SEED)
key_iter = _get_key_iter(key)
# -

# ## 3. Core Problem Definition

# +
def setup_problem(config):
    """
    Sets up the core components of the PDDS problem.
    
    Returns:
        tuple: (target_distribution, sde, log_g0, mcmc_step_size_scheduler)
    """
    # Target distribution
    target_distribution = NormalDistributionWrapper(
        mean=config["target_mean"], 
        scale=config["target_scale"], 
        dim=config["dim"], 
        is_target=True
    )
    
    # SDE for diffusion process
    scheduler = LinearScheduler(
        t_0=config["t_0"], 
        t_f=config["t_f"], 
        beta_0=config["beta_0"], 
        beta_f=config["beta_f"]
    )
    sde = SDE(scheduler, sigma=config["sigma"], dim=config["dim"])
    
    # Base potential function
    log_g0 = RatioPotential(sigma=config["sigma"], target=target_distribution)
    
    # MCMC step size scheduler (identity function in this example)
    mcmc_step_size_scheduler = lambda x: x
    
    return target_distribution, sde, log_g0, mcmc_step_size_scheduler

# Initialize core components
target_distribution, sde, log_g0, mcmc_step_size_scheduler = setup_problem(CONFIG)
# -

# ## 4. Naive Potential Approximation Evaluation

# +
def evaluate_sampler(smc_problem, num_particles, num_eval_runs=100):
    """
    Evaluates a sampler by running it multiple times and calculating statistics.
    
    Args:
        smc_problem: The SMC problem definition
        num_particles: Number of particles to use in sampling
        num_eval_runs: Number of evaluation runs
        
    Returns:
        tuple: (mean_log_Z, std_log_Z, final_samples)
    """
    # Create the evaluation sampler
    eval_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=num_particles,
            ess_threshold=0.3,
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
            density_state=0,
        )
    )
    
    # Evaluate the normalizing constant estimate over multiple runs
    log_Z = np.zeros(num_eval_runs)
    key_eval = jax.random.PRNGKey(SEED + 100)  # Different seed for evaluation
    
    for i in range(num_eval_runs):
        key_eval, subkey = jax.random.split(key_eval)
        smc_result, _ = eval_sampler(subkey)
        log_Z[i] = smc_result["log_normalising_constant"]
    
    # Get the final samples from the last run
    key_eval, subkey = jax.random.split(key_eval)
    final_samples = resampler(
        rng=subkey, 
        samples=smc_result["samples"], 
        log_weights=smc_result["log_weights"]
    )["samples"]
    
    return np.mean(log_Z), np.std(log_Z), final_samples, smc_result

def plot_distribution_comparison(samples_list, labels, title="Distribution Comparison"):
    """
    Plots KDE of samples from different methods against the target distribution.
    
    Args:
        samples_list: List of sample arrays to plot
        labels: List of labels for each set of samples
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for samples, label in zip(samples_list, labels):
        sns.kdeplot(samples[:, 0], ax=ax, label=label)
    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Initialize naive potential approximation
uncorrected_approx_potential = NaivelyApproximatedPotential(
    base_potential=log_g0, 
    dim=CONFIG["dim"], 
    nn_potential_approximator=None
)

# Set up SMC problem with naive potential
naive_smc_problem = SMCProblem(sde, uncorrected_approx_potential, CONFIG["num_steps"])

# Evaluate naive approximation
print("Evaluating naive potential approximation...")
naive_log_Z_mean, naive_log_Z_std, naive_samples, _ = evaluate_sampler(
    naive_smc_problem, 
    CONFIG["num_particles"]
)

print(f'Naive log Z estimate: {naive_log_Z_mean:.4f} ± {naive_log_Z_std:.4f}')

# Generate target samples for comparison
key, subkey = jax.random.split(key)
target_samples = target_distribution.sample(subkey, num_samples=CONFIG["num_particles"])

# Plot comparison
plot_distribution_comparison(
    [naive_samples, target_samples],
    ["Naive approximation", "Target"],
    "Naive Potential Approximation vs Target"
)
# -

# ## 5. Neural Network Potential Approximation

# +
def define_nn_potential(uncorrected_approx_potential, dim):
    """
    Defines the neural network potential approximator.
    
    Args:
        uncorrected_approx_potential: The base potential approximation
        dim: Dimension of the problem
        
    Returns:
        Function that approximates the potential
    """
    @hk.without_apply_rng
    @hk.transform
    @check_shapes("lbd: [b]", "x: [b, d]")
    def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
        # Get base approximation
        residual, density_state = uncorrected_approx_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )
        
        # Define neural network
        net = PISGRADNet(hidden_shapes=[64, 64], act='gelu', dim=dim)
        
        # Get output
        out = net(lbd, x, residual)
        return out, density_state
    
    return nn_potential_approximator

def define_loss_functions(nn_potential_approximator, sde, log_g0):
    """
    Defines the loss functions and optimization utilities for training.
    
    Returns:
        Tuple of (loss_fn, update_step, init)
    """
    # Define log probability and its gradient
    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]", "return[0]: [b]")
    def log_pi(params, lbd: Array, x: Array, density_state: int) -> tp.Tuple[Array, int]:
        reference_term, _ = sde.reference_dist.evaluate_log_density(
            x=x, density_state=0
        )
        nn_approx, density_state = nn_potential_approximator.apply(
            params, lbd, x, density_state
        )
        return nn_approx + reference_term, density_state

    grad_log_pi = jax.jit(x_gradient_stateful_parametrised(log_pi))

    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_potential_approximator.apply)(
            params, lbd, x, density_state
        )

    # Define loss functions
    @check_shapes("samples: [b, d]")
    def guidance_loss_fn(params, samples: Array, key: Key, density_state: int):
        return guidance_loss(
            key,
            sde,
            partial(grad_log_g, params),
            samples,
            density_state,
            log_g0,
            False
        )
    
    @check_shapes("samples: [b, d]")
    def dsm_loss_fn(params, samples: Array, key: Key, density_state: int):
        return dsm_loss(
            key,
            sde,
            partial(grad_log_g, params),
            samples,
            density_state,
            log_g0,
            False
        )
    
    # Choose which loss function to use
    loss_fn = guidance_loss_fn  # could also use dsm_loss_fn
    
    # Define learning rate scheduler and optimizer
    learning_rate_schedule_unlooped = optax.exponential_decay(
        init_value=CONFIG["learning_rate"], 
        transition_steps=CONFIG["lr_transition_steps"], 
        decay_rate=CONFIG["lr_decay_rate"]
    )
    learning_rate_schedule = loop_schedule(
        schedule=learning_rate_schedule_unlooped, 
        freq=CONFIG["optim_steps"]
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )
    
    # Define update step function
    @jax.jit
    @check_shapes("samples: [b, d]")
    def update_step(
        state: TrainingState, samples: Array, density_state: int
    ) -> tp.Tuple[TrainingState, int, tp.Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_value, density_state), grads = loss_and_grad_fn(
            state.params, samples, loss_key, density_state
        )
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * CONFIG["ema_decay"] + p * (1.0 - CONFIG["ema_decay"]),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1,
        )
        metrics = {"loss": loss_value, "step": state.step, "lr": learning_rate_schedule(state.step)}
        return new_state, density_state, metrics
    
    # Define initialization function
    @check_shapes("samples: [b, d]")
    def init(samples: Array, key: Key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        lbd = broadcast(jnp.array(1.0), samples)
        density_state = 0
        initial_params = nn_potential_approximator.init(
            init_rng, lbd, samples, density_state
        )
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )
    
    return loss_fn, update_step, init, optimizer

# Define NN potential approximator
nn_potential_approximator = define_nn_potential(uncorrected_approx_potential, CONFIG["dim"])

# Define loss functions and optimizers
loss_fn, update_step, init_fn, optimizer = define_loss_functions(
    nn_potential_approximator, sde, log_g0
)

# Initialize model
initial_samples = sde.reference_dist.sample(
    jax.random.PRNGKey(SEED), 
    num_samples=CONFIG["batch_size"]
)
training_state = init_fn(initial_samples, jax.random.PRNGKey(SEED))

# Print number of trainable parameters
nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
print(f"Number of parameters in neural network: {nb_params}")
# -

# ## 6. Training the Neural Network Potential

# +
def train_nn_potential(
    training_state, 
    smc_problem, 
    update_step, 
    config, 
    key
):
    """
    Trains the neural network potential approximation.
    
    Args:
        training_state: Initial training state
        smc_problem: SMC problem for generating training samples
        update_step: Function for updating the model
        config: Configuration dictionary
        key: Random key
        
    Returns:
        Final training state
    """
    # Training sampler
    training_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=config["batch_size"] * config["refresh_batch_every"],
            ess_threshold=0.3,
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        )
    )
    
    # Initial JIT compilation
    _, _ = training_sampler(rng=key, density_state=0)
    
    # Setup progress bar
    progress_bar = tqdm.tqdm(
        list(range(1, config["optim_steps"] + 1)),
        miniters=1,
        disable=False,
        desc="Training NN Potential"
    )
    
    # Train the model
    density_state_training = 0
    start_time = time.time()
    
    for step, step_key in zip(progress_bar, key_iter):
        # Generate training samples from current model
        if (step - 1) % config["refresh_batch_every"] == 0:
            jit_results, density_state_training = training_sampler(
                rng=step_key, density_state=density_state_training
            )
            sample_batches = jit_results["samples"].reshape(
                (config["refresh_batch_every"], config["batch_size"], config["dim"])
            )
        
        # Get current batch of samples
        samples = sample_batches[(step - 1) % config["refresh_batch_every"]]
        
        # Update model
        training_state, density_state_training, metrics = update_step(
            training_state, samples, density_state_training
        )
        
        # Update progress bar
        if step % 100 == 0:
            progress_bar.set_description(f"Loss: {metrics['loss']:.4f}, LR: {metrics['lr']:.6f}")
    
    end_time = time.time()
    print(f'Training complete in {end_time - start_time:.2f} seconds')
    
    return training_state

# Train the neural network potential
print("Training neural network potential approximation...")
key, subkey = jax.random.split(key)
training_state = train_nn_potential(
    training_state,
    naive_smc_problem,
    update_step,
    CONFIG,
    subkey
)
# -

# ## 7. Evaluating Improved PDDS with Neural Network Potential

# +
# Initialize the improved potential approximation using the trained network
corrected_approx_potential = NNApproximatedPotential(
    base_potential=log_g0,
    dim=CONFIG["dim"],
    nn_potential_approximator=partial(
        nn_potential_approximator.apply,
        params=training_state.params_ema
    )
)

# Set up improved SMC problem
improved_smc_problem = SMCProblem(sde, corrected_approx_potential, CONFIG["num_steps"])

# Evaluate improved model
print("Evaluating neural network potential approximation...")
nn_log_Z_mean, nn_log_Z_std, nn_samples, _ = evaluate_sampler(
    improved_smc_problem, 
    CONFIG["num_particles"]
)

print(f'PDDS with NN potential log Z estimate: {nn_log_Z_mean:.4f} ± {nn_log_Z_std:.4f}')
print(f'Improvement over naive approach: {nn_log_Z_mean - naive_log_Z_mean:.4f}')

# Plot comparison of all methods
plot_distribution_comparison(
    [naive_samples, nn_samples, target_samples],
    ["Naive approximation", "NN-improved PDDS", "Target"],
    "Comparison of Sampling Methods"
)

# Plot metrics to show the quality of each approach
metrics = {
    "Method": ["Naive Potential", "NN Potential"],
    "Log Z Mean": [naive_log_Z_mean, nn_log_Z_mean],
    "Log Z Std": [naive_log_Z_std, nn_log_Z_std]
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics["Method"]))
width = 0.35

# Plot log Z means with error bars
ax.bar(x, metrics["Log Z Mean"], width, yerr=metrics["Log Z Std"], 
       label='Log Normalizing Constant', capsize=5)

ax.set_ylabel('Log Z')
ax.set_title('Comparison of Normalizing Constant Estimates')
ax.set_xticks(x)
ax.set_xticklabels(metrics["Method"])
ax.legend()

plt.tight_layout()
plt.show()
# -

# ## 8. Summary and Conclusion

# +
# Print a summary of the experiment
print("=" * 80)
print("PDDS Sampling Experiment Summary")
print("=" * 80)
print(f"Target: 1D Gaussian with mean={CONFIG['target_mean']}, scale={CONFIG['target_scale']}")
print(f"Dimension: {CONFIG['dim']}")
print(f"Number of diffusion steps: {CONFIG['num_steps']}")
print(f"Number of particles: {CONFIG['num_particles']}")
print("-" * 80)
print("Results:")
print(f"Naive potential log Z: {naive_log_Z_mean:.4f} ± {naive_log_Z_std:.4f}")
print(f"NN potential log Z: {nn_log_Z_mean:.4f} ± {nn_log_Z_std:.4f}")
print(f"Improvement: {nn_log_Z_mean - naive_log_Z_mean:.4f}")
print("=" * 80)

"""
Simplest case - 1D shifted Gaussian distribution (mean=2.75, scale=0.25)

Training process:
Train directly on target distribution

Training parameters:
Optim steps: 10000
Discretization: 8 steps
Batch size: 300

Results:
Naive potential log Z: -27.4732 ± 2.3565
NN potential log Z: -0.0801 ± 0.4877
Improvement: 27.3931

Conclusion:
Standard PDDS is sufficient for 1D Gaussian
"""
# -


