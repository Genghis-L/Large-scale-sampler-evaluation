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

# # Gaussian Mixture Sampling - Progressive Annealing
# 
# This notebook demonstrates sampling from a challenging 2D Gaussian mixture using PDDS
# (Potential-Driven Diffusion Sampling) with a progressive annealing approach,
# gradually transitioning from a simple distribution to the complex target.

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
NUM_STEPS = 48  # Number of discretization steps for the SDE (more for progressive annealing)
NUM_PARTICLES = 2000  # Number of particles for sampling
BATCH_SIZE = 600  # Batch size for training (larger for progressive annealing)
TRAIN_STEPS_INITIAL = 4000  # Number of training steps for first stage
TRAIN_STEPS_PER_STAGE = 3000  # Number of training steps per annealing stage
ESS_THRESHOLD = 0.3  # Resampling threshold for SMC
SAVE_FIGS = True  # Whether to save figures
FIGURE_DIR = "figures/2D-hard-GM_annealing"  # Directory to save figures
NET_WIDTH = 64  # Width of neural network layers
LR_TRANSITION_STEP = 100  # Learning rate decay transition steps
LR_INIT = 1e-3  # Initial learning rate

# Annealing parameters
BETA_START = 0.3  # Starting beta value (simplest distribution)
# Create sequence of beta values from BETA_START to 1.0
NUM_START_REPEATS = 2  # Number of times to repeat the initial beta value
BETA_VALUES = [BETA_START] * NUM_START_REPEATS + list(np.repeat(np.arange(BETA_START+0.1, 1.001, 0.1), 2))

# Create figure directory if it doesn't exist
if SAVE_FIGS:
    os.makedirs(FIGURE_DIR, exist_ok=True)

print(f"Progressive annealing with beta values: {BETA_VALUES}")

# Initialize random key iterator
key = jax.random.PRNGKey(seed=0)
key_iter = _get_key_iter(key)

# Helper function for plotting results
def plot_2d_scatter(target_samples, samples, label, title=None, save_filename=None):
    """
    Plot 2D scatter plot comparing target and sampled distributions.
    
    Args:
        target_samples: Samples from the target distribution
        samples: Samples from the sampling algorithm
        label: Label for the sampling algorithm
        title: Optional title for the plot
        save_filename: Optional filename to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label=label, s=10)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"2D Gaussian Mixture (mean_scale={MEAN_SCALE})")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_filename:
        directory = os.path.dirname(save_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

# ### Stage 1: Initial Annealing with Naive Potential (β=BETA_START)

# Initialize target distributions
target_distribution = ChallengingTwoDimensionalMixture(
    mean_scale=MEAN_SCALE, dim=DIM, is_target=True
)
target_distribution_initial = ChallengingTwoDimensionalMixture(
    mean_scale=MEAN_SCALE, dim=DIM, is_target=True, beta=BETA_START
)

# Initialize SDE
scheduler = LinearScheduler(t_0=0.0, t_f=1.0, beta_0=0.001, beta_f=12.0)
sde = SDE(scheduler, sigma=SIGMA, dim=DIM)

# Initialize naive potential approximation for initial distribution
log_g0_initial = RatioPotential(sigma=SIGMA, target=target_distribution_initial)
naive_potential_initial = NaivelyApproximatedPotential(
    base_potential=log_g0_initial, dim=DIM, nn_potential_approximator=None
)

# MCMC step size scheduler (identity function - not used)
mcmc_step_size_scheduler = lambda x: x

# Initialize SMC problem with naive potential
smc_problem_initial = SMCProblem(sde, naive_potential_initial, NUM_STEPS)

# Define sampler function
naive_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem_initial,
        num_particles=NUM_PARTICLES,
        ess_threshold=ESS_THRESHOLD,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        density_state=0,
    )
)

print(f"Initial Stage: Evaluating naive potential approximation at β={BETA_START}...")
# Evaluate normalizing constant
log_Z_naive = np.zeros(100)
for i in tqdm.trange(100, disable=False):
    key, subkey = jax.random.split(key)
    smc_result_naive, _ = naive_sampler(subkey)
    log_Z_naive[i] = smc_result_naive["log_normalising_constant"]

print(f'Naive log Z estimate (β={BETA_START}): {np.mean(log_Z_naive):.4f}')

# Get samples for visualization
n_plot_samples = NUM_PARTICLES
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples_initial = target_distribution_initial.sample(subkey2, num_samples=n_plot_samples)
naive_samples = resampler(
    rng=subkey3, 
    samples=smc_result_naive["samples"], 
    log_weights=smc_result_naive["log_weights"]
)["samples"]

# Visualize naive sampling results for initial distribution
plot_2d_scatter(
    target_samples_initial,
    naive_samples,
    label="Naive Approx.",
    title=f"Initial Distribution (β={BETA_START}) - Naive Approximation",
    save_filename=f"{FIGURE_DIR}/initial_naive_sampling_beta_{BETA_START:.1f}.png"
)

# ### Neural Network for Progressive Annealing

# Define a function to create potential approximator functions with the current naive potential
def create_potential_approximator(naive_potential):
    @hk.without_apply_rng
    @hk.transform
    @check_shapes("lbd: [b]", "x: [b, d]")
    def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
        # The residual will be computed by the provided naive potential
        residual, density_state = naive_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )
        
        # Define neural network
        net = PISGRADNet(hidden_shapes=[NET_WIDTH, NET_WIDTH], act='gelu', dim=DIM)
        
        # Get network output
        out = net(lbd, x, residual)
        
        return out, density_state
    
    return nn_potential_approximator

# Create the initial neural network potential approximator
nn_potential_approximator = create_potential_approximator(naive_potential_initial)

# Define a function to create gradient functions for any nn_potential_approximator
def create_grad_log_g(nn_pot_approx):
    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_pot_approx.apply)(
            params, lbd, x, density_state
        )
    
    return grad_log_g

# Create initial gradient function
grad_log_g = create_grad_log_g(nn_potential_approximator)

# Define loss function (guidance loss)
@check_shapes("samples: [b, d]")
def guidance_loss_fn(params, samples: Array, key: Key, density_state: int, log_g0, curr_grad_log_g):
    return guidance_loss(
        key,
        sde,
        partial(curr_grad_log_g, params),
        samples,
        density_state,
        log_g0,
        False
    )

# Define learning rate schedule
learning_rate_schedule_unlooped = optax.exponential_decay(
    init_value=LR_INIT, 
    transition_steps=LR_TRANSITION_STEP, 
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
    state: TrainingState, samples: Array, density_state: int, loss_fn
) -> tp.Tuple[TrainingState, int, tp.Mapping]:
    # Split random key
    new_key, loss_key = jax.random.split(state.key)
    
    # Compute loss and gradients
    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
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
def init_model(samples: Array, key: Key) -> TrainingState:
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

# Initialize training state with samples from the reference distribution
initial_samples = sde.reference_dist.sample(
    jax.random.PRNGKey(seed=0), num_samples=BATCH_SIZE
)
training_state = init_model(initial_samples, jax.random.PRNGKey(seed=0))

# Print number of parameters
nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
print(f"Number of trainable parameters: {nb_params}")

# Define training control parameters
refresh_batch_every = 100  # Number of steps before refreshing the batch of samples

# ### Progressive Annealing: Training Through Multiple Beta Values

# Helper function to create a snapshot of the current potential
def create_snapshot_potential_approximator(params, nn_pot_approx):
    @check_shapes("lbd: [b]", "x: [b, d]", "return[0]: [b]")
    def snapshot_potential_approximator(lbd: Array, x: Array, density_state: int):
        # Use the fixed snapshot of parameters
        return nn_pot_approx.apply(
            params,  # These parameters won't change when training_state is updated
            lbd, 
            x,
            density_state
        )

    return snapshot_potential_approximator

# Track normalization constant estimates across annealing stages
log_Z_estimates = {BETA_START: np.mean(log_Z_naive)}

# Create a list to track beta values for validation
beta_validation_values = list(np.arange(BETA_START+0.1, 0.99, 0.1))
next_valid_idx = 0

# Initialize a list to store samples from each beta stage for visualization
beta_samples = {BETA_START: naive_samples}

# Progressive annealing loop
print("\n=== Starting Progressive Annealing ===\n")
for beta_idx, beta in enumerate(BETA_VALUES[NUM_START_REPEATS:], NUM_START_REPEATS):
    # Determine number of training steps for this stage
    optim_steps = TRAIN_STEPS_PER_STAGE
    
    print(f"\n--- Annealing Stage {beta_idx-NUM_START_REPEATS+1}: β={beta:.1f} ---")
    
    # Create target distribution for current beta
    target_distribution_current = ChallengingTwoDimensionalMixture(
        mean_scale=MEAN_SCALE, dim=DIM, is_target=True, beta=beta
    )
    
    # Create potential for current beta
    log_g0_current = RatioPotential(sigma=SIGMA, target=target_distribution_current)
    
    # Update current naive potential for this beta value
    naive_potential_current = NaivelyApproximatedPotential(
        base_potential=log_g0_current, dim=DIM, nn_potential_approximator=None
    )
    
    # Update the neural network potential approximator for the current beta
    new_nn_potential_approximator = create_potential_approximator(naive_potential_current)
    
    # Update the gradient function for the current neural net
    grad_log_g = create_grad_log_g(new_nn_potential_approximator)
    
    # For the snapshot, we still use the previous nn_potential_approximator
    # Create a snapshot of the current model to use as base potential
    snapshot_params = training_state.params_ema
    snapshot_approximator = create_snapshot_potential_approximator(snapshot_params, new_nn_potential_approximator)
    
    # Create the NNApproximatedPotential with the snapshot
    snapshot_potential = NNApproximatedPotential(
        base_potential=log_g0_current,
        dim=DIM,
        nn_potential_approximator=snapshot_approximator
    )
    
    # Create SMC problem for training
    smc_problem_current = SMCProblem(sde, snapshot_potential, NUM_STEPS)
    
    # Set up training sampler for current beta
    training_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem_current,
            num_particles=BATCH_SIZE * refresh_batch_every,
            ess_threshold=ESS_THRESHOLD,
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        )
    )
    
    # Initial JIT compilation
    _, _ = training_sampler(rng=key, density_state=0)
    
    # Train neural network for current beta
    print(f"Training neural network for β={beta:.1f}...")
    progress_bar = tqdm.tqdm(
        list(range(1, optim_steps + 1)),
        miniters=1,
        disable=False,
    )
    
    # Initialize/reset density state
    density_state_training = 0
    
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
        
        # Define loss function for this stage
        current_loss_fn = partial(guidance_loss_fn, log_g0=log_g0_current, curr_grad_log_g=grad_log_g)
        
        # Update model
        training_state, density_state_training, metrics = update_step(
            training_state, samples, density_state_training, current_loss_fn
        )
    
        # Update metrics
        metrics["lr"] = learning_rate_schedule(training_state.step)
    
        # Update progress bar
        if step % 100 == 0:
            progress_bar.set_description(f"Loss: {metrics['loss']:.4f}")
    
    train_time = time.time() - start_time
    print(f'Training for β={beta:.1f} complete, time: {train_time:.2f}s')
    
    # Create neural network potential approximation for evaluation
    nn_potential_current = NNApproximatedPotential(
        base_potential=log_g0_current,
        dim=DIM,
        nn_potential_approximator=partial(
            new_nn_potential_approximator.apply,
            params=training_state.params_ema
        )
    )
    
    # Initialize SMC problem with neural network potential
    smc_problem_nn_current = SMCProblem(sde, nn_potential_current, NUM_STEPS)
    
    # Define sampler function
    nn_sampler_current = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem_nn_current,
            num_particles=NUM_PARTICLES,
            ess_threshold=ESS_THRESHOLD,
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
            density_state=0,
        )
    )
    
    # Check if this is a validation beta or the final beta
    should_validate = (next_valid_idx < len(beta_validation_values) and 
                      beta >= beta_validation_values[next_valid_idx] - 1e-10) or (
                      beta_idx == len(BETA_VALUES) - 1)
    
    if should_validate:
        # Determine number of samples for normalizing constant estimation
        n_sample_for_z = 100 if beta_idx == len(BETA_VALUES) - 1 else 5
        
        print(f"Evaluating PDDS sampler at β={beta:.1f}...")
        # Evaluate normalizing constant
        log_Z_current = np.zeros(n_sample_for_z)
        for i in tqdm.trange(n_sample_for_z, disable=False):
            key, subkey = jax.random.split(key)
            smc_result_current, _ = nn_sampler_current(subkey)
            log_Z_current[i] = smc_result_current["log_normalising_constant"]
        
        log_Z_mean = np.mean(log_Z_current)
        log_Z_estimates[beta] = log_Z_mean
        
        if beta_idx == len(BETA_VALUES) - 1:
            print(f'Final PDDS log Z estimate (β={beta:.1f}): {log_Z_mean:.4f}')
        else:
            print(f'PDDS log Z estimate (β={beta:.1f}): {log_Z_mean:.4f}')
        
        # Get samples for visualization
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        target_samples_current = target_distribution_current.sample(subkey2, num_samples=n_plot_samples)
        nn_samples_current = resampler(
            rng=subkey3, 
            samples=smc_result_current["samples"], 
            log_weights=smc_result_current["log_weights"]
        )["samples"]
        
        # Store samples for later visualization
        beta_samples[beta] = nn_samples_current
        
        # Visualize neural network sampling results
        plot_2d_scatter(
            target_samples_current,
            nn_samples_current,
            label="PDDS",
            title=f"Distribution (β={beta:.1f}) - PDDS Sampling",
            save_filename=f"{FIGURE_DIR}/pdds_sampling_beta_{beta:.1f}.png"
        )
        
        next_valid_idx += 1

# ### Visualize Progression Across Annealing Stages

# Plot the progression of annealing
sorted_betas = sorted(beta_samples.keys())
if len(sorted_betas) >= 3:
    num_plots = min(5, len(sorted_betas))
    selected_betas = sorted_betas[:1] + sorted_betas[1:num_plots-1:max(1, (len(sorted_betas)-2)//(num_plots-2))] + sorted_betas[-1:]
    
    plt.figure(figsize=(15, 5))
    for i, beta in enumerate(selected_betas):
        plt.subplot(1, len(selected_betas), i+1)
        plt.scatter(beta_samples[beta][:, 0], beta_samples[beta][:, 1], alpha=0.5, s=5)
        plt.title(f"β={beta:.1f}")
        plt.axis('equal')
        plt.grid(True)
    
    plt.suptitle(f"Progression of Samples Across Annealing Stages (mean_scale={MEAN_SCALE})")
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(f"{FIGURE_DIR}/annealing_progression.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Plot the estimated normalizing constants
plt.figure(figsize=(10, 5))
betas = sorted(log_Z_estimates.keys())
log_Zs = [log_Z_estimates[beta] for beta in betas]
plt.plot(betas, log_Zs, 'o-')
plt.xlabel("β value")
plt.ylabel("Log normalizing constant")
plt.title("Estimated Log Normalizing Constants Across Annealing Stages")
plt.grid(True)
if SAVE_FIGS:
    plt.savefig(f"{FIGURE_DIR}/log_Z_progression.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\n=== Progressive Annealing Complete ===\n")
print(f"Final log normalizing constant estimate: {log_Z_estimates[1.0]:.4f}")
print(f"Results saved to {FIGURE_DIR}/")

print("Experiment complete!") 