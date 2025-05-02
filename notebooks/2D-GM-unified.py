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

# # Gaussian Mixture Sampling - Basic Version
# 
# This notebook demonstrates direct sampling from a challenging 2D Gaussian mixture using PDDS 
# (Potential-Driven Diffusion Sampling) without annealing.

# ### Setup and Imports

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set available GPU

import time
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from functools import partial
from jaxtyping import PRNGKeyArray as Key, Array
import typing as tp
from check_shapes import check_shapes

# PDDS imports
from pdds.sde import SDE, guidance_loss, dsm_loss
from pdds.smc_problem import SMCProblem
from pdds.potentials import NNApproximatedPotential
from pdds.utils.shaping import broadcast
from pdds.utils.jax import _get_key_iter, x_gradient_stateful_parametrised
from pdds.utils.lr_schedules import loop_schedule
from pdds.ml_tools.state import TrainingState
from pdds.smc_loops import fast_outer_loop_smc
from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture
from pdds.sde import LinearScheduler, SDE
from pdds.potentials import RatioPotential, NaivelyApproximatedPotential, NNApproximatedPotential
from pdds.nn_models.mlp import PISGRADNet
from pdds.resampling import resampler

# ### Configuration Parameters

# Global parameters
MEAN_SCALE = 3.0  # Scale factor for the mixture components
DIM = 2  # Dimension of the problem
SIGMA = 1.0  # Diffusion strength
NUM_STEPS = 10  # Number of discretization steps for the SDE
NUM_PARTICLES = 2000  # Number of particles for sampling
BATCH_SIZE = 300  # Batch size for training
TRAIN_STEPS = 10000  # Number of training steps
ESS_THRESHOLD = 0.3  # Resampling threshold for SMC
SAVE_FIGS = True  # Whether to save figures
FIGURE_DIR = "figures/basic"  # Directory to save figures

# Create figure directory if it doesn't exist
if SAVE_FIGS:
    os.makedirs(FIGURE_DIR, exist_ok=True)

# Initialize random key iterator
key = jax.random.PRNGKey(seed=0)
key_iter = _get_key_iter(key)

# ### Step 1: Naive Sampling with Potential Approximation

# Initialize target distribution
target_distribution = ChallengingTwoDimensionalMixture(
    mean_scale=MEAN_SCALE, dim=DIM, is_target=True
)

# Initialize SDE
scheduler = LinearScheduler(t_0=0.0, t_f=1.0, beta_0=0.001, beta_f=12.0)
sde = SDE(scheduler, sigma=SIGMA, dim=DIM)

# Initialize naive potential approximation
log_g0 = RatioPotential(sigma=SIGMA, target=target_distribution)
naive_potential = NaivelyApproximatedPotential(
    base_potential=log_g0, dim=DIM, nn_potential_approximator=None
)

# Initialize SMC problem with naive potential
smc_problem = SMCProblem(sde, naive_potential, NUM_STEPS)

# MCMC step size scheduler (identity function - not used)
mcmc_step_size_scheduler = lambda x: x

# Define sampler function
naive_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem,
        num_particles=NUM_PARTICLES,
        ess_threshold=ESS_THRESHOLD,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        density_state=0,
    )
)

print("Evaluating naive potential approximation...")
# Evaluate normalizing constant
log_Z_naive = np.zeros(100)
for i in tqdm.trange(100, disable=False):
    key, subkey = jax.random.split(key)
    smc_result_naive, _ = naive_sampler(subkey)
    log_Z_naive[i] = smc_result_naive["log_normalising_constant"]

print(f'Naive log Z estimate: {np.mean(log_Z_naive):.4f}')

# Get samples for visualization
n_plot_samples = NUM_PARTICLES
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
naive_samples = resampler(
    rng=subkey3, 
    samples=smc_result_naive["samples"], 
    log_weights=smc_result_naive["log_weights"]
)["samples"]

# Visualize naive sampling results
plt.figure(figsize=(8, 6))
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
plt.scatter(naive_samples[:, 0], naive_samples[:, 1], alpha=0.5, label="Naive Approx.", s=10)
plt.title(f"2D Gaussian Mixture (mean_scale={MEAN_SCALE}) - Naive Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
# if SAVE_FIGS:
#     plt.savefig(f"{FIGURE_DIR}/naive_sampling.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ### Step 2: Neural Network Potential Approximation

# Define neural network potential approximator
@hk.without_apply_rng
@hk.transform
@check_shapes("lbd: [b]", "x: [b, d]")
def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
    # Get residual from naive approximation
    residual, density_state = naive_potential.approx_log_gt(
        lbd=lbd, x=x, density_state=density_state
    )
    
    # Define neural network
    net = PISGRADNet(hidden_shapes=[64, 64], act='gelu', dim=DIM)
    
    # Get network output
    out = net(lbd, x, residual)
    
    return out, density_state

# Define gradient function for the neural network
@jax.jit
@check_shapes("lbd: [b]", "x: [b, d]")
def grad_log_g(params, lbd: Array, x: Array, density_state: int):
    return x_gradient_stateful_parametrised(nn_potential_approximator.apply)(
        params, lbd, x, density_state
    )

# Define loss function (guidance loss)
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

# Define learning rate schedule
learning_rate_schedule_unlooped = optax.exponential_decay(
    init_value=1e-3, 
    transition_steps=50, 
    decay_rate=0.95
)
learning_rate_schedule = loop_schedule(
    schedule=learning_rate_schedule_unlooped, 
    freq=10000
)

# Define optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

# Define model update function
@jax.jit
@check_shapes("samples: [b, d]")
def update_step(
    state: TrainingState, samples: Array, density_state: int
) -> tp.Tuple[TrainingState, int, tp.Mapping]:
    # Split random key
    new_key, loss_key = jax.random.split(state.key)
    
    # Compute loss and gradients
    loss_and_grad_fn = jax.value_and_grad(guidance_loss_fn, has_aux=True)
    (loss_value, density_state), grads = loss_and_grad_fn(
        state.params, samples, loss_key, density_state
    )
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    
    # Update EMA parameters
    new_params_ema = jax.tree_util.tree_map(
        lambda p_ema, p: p_ema * 0.999 + p * (1.0 - 0.999),
        state.params_ema,
        new_params,
    )
    
    # Create new state
    new_state = TrainingState(
        params=new_params,
        params_ema=new_params_ema,
        opt_state=new_opt_state,
        key=new_key,
        step=state.step + 1,
    )
    
    # Return metrics
    metrics = {"loss": loss_value, "step": state.step}
    
    return new_state, density_state, metrics

# Initialize model
@check_shapes("samples: [b, d]")
def init(samples: Array, key: Key) -> TrainingState:
    # Split random key
    key, init_rng = jax.random.split(key)
    
    # Initialize parameters
    lbd = broadcast(jnp.array(1.0), samples)
    density_state = 0
    initial_params = nn_potential_approximator.init(
        init_rng, lbd, samples, density_state
    )
    
    # Initialize optimizer state
    initial_opt_state = optimizer.init(initial_params)
    
    # Return initial state
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )

# Initialize training state
initial_samples = sde.reference_dist.sample(
    jax.random.PRNGKey(seed=0), num_samples=BATCH_SIZE
)
training_state = init(initial_samples, jax.random.PRNGKey(seed=0))

# Print number of parameters
nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
print(f"Number of trainable parameters: {nb_params}")

# ### Step 3: Train Neural Network Potential Approximation

# Initialize training variables
density_state_training = 0
refresh_batch_every = 100

# Set up training sampler
training_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem,
        num_particles=BATCH_SIZE * refresh_batch_every,
        ess_threshold=ESS_THRESHOLD,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
    )
)

# Initial JIT compilation
_, _ = training_sampler(rng=key, density_state=0)

# Train neural network
print("Training neural network potential approximation...")
progress_bar = tqdm.tqdm(
    list(range(1, TRAIN_STEPS + 1)),
    miniters=1,
    disable=False,
)

start_time = time.time()
for step, key in zip(progress_bar, key_iter):
    # Generate new samples every 'refresh_batch_every' steps
    if (step - 1) % refresh_batch_every == 0:
        jit_results, density_state_training = training_sampler(
            rng=key, density_state=density_state_training
        )
        sample_batches = jit_results["samples"].reshape(
            (refresh_batch_every, BATCH_SIZE, DIM)
        )

    # Get current batch of samples
    samples = sample_batches[(step - 1) % refresh_batch_every]
    
    # Update model
    training_state, density_state_training, metrics = update_step(
        training_state, samples, density_state_training
    )

    # Update metrics
    metrics["lr"] = learning_rate_schedule(training_state.step)

    # Update progress bar
    if step % 100 == 0:
        progress_bar.set_description(f"Loss: {metrics['loss']:.4f}")

train_time = time.time() - start_time
print(f'Training complete, time: {train_time:.2f}s')

# ### Step 4: Evaluate Trained Model

# Create neural network potential approximation
nn_potential = NNApproximatedPotential(
    base_potential=log_g0,
    dim=DIM,
    nn_potential_approximator=partial(
        nn_potential_approximator.apply,
        params=training_state.params_ema
    )
)

# Initialize SMC problem with neural network potential
smc_problem_nn = SMCProblem(sde, nn_potential, NUM_STEPS)

# Define sampler function
nn_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem_nn,
        num_particles=NUM_PARTICLES,
        ess_threshold=ESS_THRESHOLD,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        density_state=0,
    )
)

print("Evaluating neural network potential approximation...")
# Evaluate normalizing constant
log_Z_nn = np.zeros(100)
for i in tqdm.trange(100, disable=False):
    key, subkey = jax.random.split(key)
    smc_result_nn, _ = nn_sampler(subkey)
    log_Z_nn[i] = smc_result_nn["log_normalising_constant"]

print(f'PDDS log Z estimate: {np.mean(log_Z_nn):.4f}')

# Get samples for visualization
n_plot_samples = NUM_PARTICLES
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
nn_samples = resampler(
    rng=subkey3, 
    samples=smc_result_nn["samples"], 
    log_weights=smc_result_nn["log_weights"]
)["samples"]

# Visualize neural network sampling results
plt.figure(figsize=(8, 6))
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
plt.scatter(nn_samples[:, 0], nn_samples[:, 1], alpha=0.5, label="PDDS", s=10)
plt.title(f"2D Gaussian Mixture (mean_scale={MEAN_SCALE}) - PDDS Sampling")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
# if SAVE_FIGS:
#     plt.savefig(f"{FIGURE_DIR}/pdds_sampling.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Plot comparison of sampling methods
plt.figure(figsize=(12, 5))

# Left: Naive Approximation
plt.subplot(1, 2, 1)
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
plt.scatter(naive_samples[:, 0], naive_samples[:, 1], alpha=0.5, label="Naive Approx.", s=10)
plt.title(f"Naive Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')

# Right: PDDS
plt.subplot(1, 2, 2)
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
plt.scatter(nn_samples[:, 0], nn_samples[:, 1], alpha=0.5, label="PDDS", s=10)
plt.title(f"PDDS Sampling")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.suptitle(f"Comparison of Sampling Methods (mean_scale={MEAN_SCALE})")
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig(f"{FIGURE_DIR}/comparison.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Experiment complete!") 