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

# # Hard 2D Gaussian Mixture with Annealing: PDDS

# ## 1. Setup and Imports

# +
# # %load_ext autoreload
# # %autoreload 2

import os
import time
import tqdm
import typing as tp
from functools import partial

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
from pdds.distributions import (
    NormalDistributionWrapper, 
    ChallengingTwoDimensionalMixture,
    NormalDistribution
)
from pdds.nn_models.mlp import PISGRADNet
from pdds.resampling import resampler

###############################################################################
# Patch NormalDistribution.__init__ **including** the check_shapes wrapper
###############################################################################
import jax.numpy as jnp
from pdds.distributions import NormalDistribution
from chex import assert_axis_dimension
from jaxtyping import Array
from check_shapes import check_shapes                # keep shape checking

@check_shapes("mean: [b, d]")
def _safe_init(self,
               mean: Array,
               scale,
               dim: int = 1,
               is_target: bool = False):
    """
    Replacement for NormalDistribution.__init__ that
    does **not** evaluate `jnp.any(scale <= 0)` in Python control-flow.
    """
    # ------------------------------------------------------------------ #
    #  Re-implement the constructor, but validate only **static** values #
    # ------------------------------------------------------------------ #
    from pdds.distributions import Distribution        # local import
    Distribution.__init__(self, dim, is_target)

    assert_axis_dimension(mean, 1, dim)

    # Validate *only* scalar (Python) numbers.
    if isinstance(scale, (float, int)) and scale <= 0:
        raise ValueError("Scale must be positive")

    # For JAX arrays we just trust the user; if somebody actually passes
    # a negative scale a later sqrt/LogPDF will still fail at run-time.
    self._mean = mean
    self._scale = scale
    self._cov_matrix = self._scale ** 2 * jnp.eye(self.dim)

# Replace the *wrapper* that every import sees
NormalDistribution.__init__ = _safe_init
###############################################################################
# -

# ## 2. Configuration

# +
def get_config():
    """
    Return configuration parameters for the experiment.
    
    This centralized configuration function makes it easy to adjust parameters
    for different experimental settings.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    config = {
        # Environment settings
        "cuda_visible_devices": "0",
        
        # Problem dimension settings
        "dim": 2,                    # Dimensionality of the problem
        "mean_scale": 3.0,           # Scale parameter for the Gaussian mixture (higher = more separated modes)
        "sigma": 1.0,                # Noise scale parameter
        
        # SDE settings
        "t_0": 0.0,                  # Initial time
        "t_f": 1.0,                  # Final time
        "num_steps": 48,             # Number of discretization steps
        
        # SMC settings
        "num_particles": 2000,       # Number of particles for sampling
        "ess_threshold": 0.3,        # Effective sample size threshold for resampling
        
        # Training settings
        "beta_start": 0.3,           # Initial annealing parameter
        "net_width": 64,             # Width of neural network layers
        "lr_transition_step": 100,   # Learning rate decay step size
        "lr_init": 1e-3,             # Initial learning rate
        "batch_size": 600,           # Batch size for training
        "refresh_batch_every": 100,  # Number of steps before refreshing training batch
        "optim_step_start": 4000,    # Number of optimization steps for initial training
        
        # Continued training settings
        "batch_size_continued": 1000,      # Batch size for continued training
        "optim_step_continued_total": 30000, # Total steps for continued training
        "num_start_repeats": 2,            # Number of times to repeat initial training
        
        # Evaluation settings
        "n_eval_samples": 100,       # Number of samples for evaluation
        "num_particles_eval": 4000,  # Number of particles for evaluation
        
        # Visualization
        "figtype": 2,                # Figure type for visualization
    }
    return config

# Set configuration
config = get_config()

# Set id of available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_visible_devices"]
# -

# ## 3. Visualization Functions

# +
def plot_simplified_visualization(target_samples, samples, label, beta=None, mean_scale=None):
    """
    Create a simplified visualization with 2D scatter plot and 1D marginals.
    
    Args:
        target_samples: Samples from target distribution
        samples: Samples from approximation
        label: Label for the approximation
        beta: Beta value for title
        mean_scale: Mean scale for title
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 2D scatter
    axs[0].scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
    axs[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, label=label, s=10)
    
    # Set title with beta and/or mean_scale if provided
    beta_str = f" (β={beta:.1f})" if beta is not None else ""
    title = f"Hard 2D Gaussian Mixture{beta_str}"
    if mean_scale is not None:
        title += f" (mean_scale={mean_scale})"
    axs[0].set_title(title)
    
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')
    
    # X-axis marginal
    sns.kdeplot(target_samples[:, 0], label="Target", ax=axs[1], alpha=0.7)
    sns.kdeplot(samples[:, 0], label=label, ax=axs[1], alpha=0.7)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Density")
    axs[1].set_title(f"1D Marginal Distribution (x-axis){beta_str}")
    axs[1].grid(True)
    axs[1].legend()
    
    # Y-axis marginal
    sns.kdeplot(target_samples[:, 1], label="Target", ax=axs[2], alpha=0.7)
    sns.kdeplot(samples[:, 1], label=label, ax=axs[2], alpha=0.7)
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("Density")
    axs[2].set_title(f"1D Marginal Distribution (y-axis){beta_str}")
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

# Replace existing plotting functions
def plot_2d_scatter(target_samples, final_samples, label, title=None, save_filename=None, mean_scale=None):
    """
    Plot scatter comparison between target and sampled distributions (simplified version).
    """
    plot_simplified_visualization(target_samples, final_samples, label, None, mean_scale)

def plot_ess_over_time(ess_values, title=None, save_filename=None):
    """Removed - ESS plots no longer needed"""
    pass

def plot_mcmc_acceptance_rates(acceptance_rates, title=None, save_filename=None):
    """Removed - MCMC acceptance rate plots no longer needed"""
    pass

def plot_1d_marginals(target_samples, sampled_samples, label, beta=None, save_filename_prefix=None):
    """
    Plot 1D marginal distributions (simplified version).
    """
    plot_simplified_visualization(target_samples, sampled_samples, label, beta)

def plot_combined_visualizations(target_samples, final_samples, label, beta=None, ess_values=None, 
                               acceptance_rates=None, mean_scale=None, save_filename_prefix=None):
    """
    Create a simplified visualization (replaced comprehensive visualization).
    """
    plot_simplified_visualization(target_samples, final_samples, label, beta, mean_scale)
# -

# ## 4. Model Definition and Training Functions

# +
def create_nn_potential_approximator(dim, net_width, base_potential):
    """
    Create a neural network potential approximator.
    
    Args:
        dim: Problem dimension
        net_width: Width of neural network layers
        base_potential: Base potential function to approximate
        
    Returns:
        Neural network potential approximator function
    """
    @hk.without_apply_rng
    @hk.transform
    @check_shapes("lbd: [b]", "x: [b, d]")
    def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
        std_trick = False
        std = None
        residual, density_state = base_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )

        net = PISGRADNet(hidden_shapes=[net_width, net_width], act='gelu', dim=dim)
        out = net(lbd, x, residual)

        if std_trick:
            out = out / (std + 1e-3)

        return out, density_state
    
    return nn_potential_approximator

def create_loss_fn(sde, nn_potential_approximator, base_potential, use_guidance=True):
    """
    Create loss function for training the potential approximator.
    
    Args:
        sde: SDE object
        nn_potential_approximator: Neural network potential approximator
        base_potential: Base potential function
        use_guidance: Whether to use guidance loss (True) or DSM loss (False)
        
    Returns:
        Loss function for training
    """
    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_potential_approximator.apply)(
            params, lbd, x, density_state
        )

    # Define guidance loss function
    @check_shapes("samples: [b, d]")
    def guidance_loss_fn(params, samples: Array, key: Key, density_state: int):
        return guidance_loss(
            key,
            sde,
            partial(grad_log_g, params),
            samples,
            density_state,
            base_potential,
            False
        )
    
    # Define denoising score matching loss function
    @check_shapes("samples: [b, d]")
    def dsm_loss_fn(params, samples: Array, key: Key, density_state: int):
        return dsm_loss(
            key,
            sde,
            partial(grad_log_g, params),
            samples,
            density_state,
            base_potential,
            False
        )

    return guidance_loss_fn if use_guidance else dsm_loss_fn

def create_optimizer(lr_init, lr_transition_step):
    """
    Create optimizer with learning rate schedule.
    
    Args:
        lr_init: Initial learning rate
        lr_transition_step: Step size for learning rate decay
        
    Returns:
        tuple: (optimizer, learning_rate_schedule)
    """
    learning_rate_schedule_unlooped = optax.exponential_decay(
        init_value=lr_init, 
        transition_steps=lr_transition_step, 
        decay_rate=0.95
    )
    learning_rate_schedule = loop_schedule(
        schedule=learning_rate_schedule_unlooped, freq=10000
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )
    return optimizer, learning_rate_schedule

def create_update_step(loss_fn, optimizer, ema_decay=0.999):
    """
    Create function for updating model parameters.
    
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer for parameter updates
        ema_decay: Decay rate for exponential moving average of parameters
        
    Returns:
        Update step function
    """
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
            lambda p_ema, p: p_ema * ema_decay + p * (1.0 - ema_decay),
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
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, density_state, metrics
    
    return update_step

def init_model(nn_potential_approximator, optimizer, samples, key):
    """
    Initialize the neural network model.
    
    Args:
        nn_potential_approximator: Neural network potential approximator
        optimizer: Optimizer for parameter updates
        samples: Initial samples for initialization
        key: Random key
        
    Returns:
        TrainingState: Initial training state
    """
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

def create_snapshot_potential_approximator(nn_potential_approximator, params):
    """
    Create a potential approximator with fixed parameters.
    
    This is useful for freezing the model at a certain state while
    continuing to train another version.
    
    Args:
        nn_potential_approximator: Neural network potential approximator
        params: Parameters to fix
        
    Returns:
        Snapshot potential approximator function
    """
    @check_shapes("lbd: [b]", "x: [b, d]", "return[0]: [b]")
    def snapshot_potential_approximator(lbd: Array, x: Array, density_state: int):
        # Use the fixed snapshot of parameters
        return nn_potential_approximator.apply(
            params,  # These parameters won't change when training_state is updated
            lbd,
            x,
            density_state
        )

    return snapshot_potential_approximator

def train_model(training_state, update_step, smc_problem, batch_size, refresh_batch_every, 
                optim_steps, learning_rate_schedule, dim):
    """
    Train the neural network model.
    
    Args:
        training_state: Initial training state
        update_step: Update step function
        smc_problem: SMC problem for sampling
        batch_size: Batch size for training
        refresh_batch_every: Number of steps before refreshing training batch
        optim_steps: Number of optimization steps
        learning_rate_schedule: Learning rate schedule
        dim: Problem dimension
        
    Returns:
        Updated training state
    """
    density_state_training = 0
    
    # Initial sampler for training samples
    training_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=batch_size * refresh_batch_every,
            ess_threshold=config["ess_threshold"],
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=lambda x: x,
        )
    )
    
    # Initial jit compilation
    key = training_state.key
    _, _ = training_sampler(rng=key, density_state=0)
    
    # Create key iterator
    key_iter = _get_key_iter(key)
    
    progress_bar = tqdm.tqdm(
        list(range(1, optim_steps + 1)),
        miniters=1,
        disable=False,
        desc=f"Training NN potential ({optim_steps} steps)"
    )
    
    start_time = time.time()
    for step, key in zip(progress_bar, key_iter):
        # Generate samples for potential approximation training
        if (step - 1) % refresh_batch_every == 0:  # refresh samples after every 'epoch'
            jit_results, density_state_training = training_sampler(
                rng=key, density_state=density_state_training
            )
            sample_batches = jit_results["samples"].reshape(
                (refresh_batch_every, batch_size, dim)
            )
            
        samples = sample_batches[(step - 1) % refresh_batch_every]
        training_state, density_state_training, metrics = update_step(
            training_state, samples, density_state_training
        )
            
        metrics["lr"] = learning_rate_schedule(training_state.step)
            
        if step % 100 == 0:
            progress_bar.set_description(f"Training loss {metrics['loss']:.4f}, lr {metrics['lr']:.6f}")
            
    end_time = time.time()
    print(f'Training complete in {end_time - start_time:.2f}s')
    
    return training_state
# -

# ## 5. Evaluation Functions

# +
def evaluate_potential(sde, potential, num_particles, ess_threshold, num_steps, n_samples=100, num_mcmc_steps=0):
    """
    Evaluate a potential by estimating the log normalizing constant.
    
    Args:
        sde: SDE object
        potential: Potential function
        num_particles: Number of particles to use
        ess_threshold: ESS threshold for resampling
        num_steps: Number of discretization steps
        n_samples: Number of samples to average over
        num_mcmc_steps: Number of MCMC steps to apply after resampling
    
    Returns:
        Tuple of (log_Z, samples, key, ess_values, acceptance_rates)
    """
    # Create problem and sampler
    smc_problem = SMCProblem(sde, potential, num_steps)
    
    # Fast sampler
    eval_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            num_mcmc_steps=num_mcmc_steps,
            mcmc_step_size_scheduler=lambda x: x,  # Identity function
            density_state=0,
        )
    )
    
    # Evaluate the normalising constant estimate averaging over n_samples seeds
    key = jax.random.PRNGKey(0)
    log_Z = np.zeros(n_samples)
    samples = None
    ess_values = None
    acceptance_rates = None
    
    for i in tqdm.trange(n_samples, desc=f"Evaluating potential ({n_samples} runs)", disable=False):
        key, subkey = jax.random.split(key)
        smc_result, _ = eval_sampler(subkey)
        log_Z[i] = smc_result["log_normalising_constant"]
        if i == 0:  # Save first sample for visualization
            samples = smc_result
            # Extract ESS and acceptance rates if available
            if "ess_history" in smc_result:
                ess_values = np.array(smc_result["ess_history"])
            if "acceptance_rates" in smc_result:
                acceptance_rates = np.array(smc_result["acceptance_rates"])
    
    log_Z_mean = np.mean(log_Z)
    log_Z_std = np.std(log_Z)
    print(f"Log Z estimate: {log_Z_mean:.4f} ± {log_Z_std:.4f}")
    
    return log_Z_mean, samples, key, ess_values, acceptance_rates
# -

# ## 6. Main Experiment

def main():
    """
    Main experiment function implementing annealed PDDS training.
    
    This experiment evaluates PDDS (PDDS) with annealing
    for sampling from a challenging 2D Gaussian mixture distribution.
    
    The experiment flows through these stages:
    1. Evaluate naive potential approximation at starting beta
    2. Train NN potential approximation at starting beta
    3. Gradually increase beta (annealing) and continue training
    4. Evaluate final performance
    """
    # Extract configuration parameters
    dim = config["dim"]
    mean_scale = config["mean_scale"]
    sigma = config["sigma"]
    t_0 = config["t_0"]
    t_f = config["t_f"]
    num_steps = config["num_steps"]
    num_particles = config["num_particles"]
    beta_start = config["beta_start"] 
    net_width = config["net_width"]
    lr_transition_step = config["lr_transition_step"]
    lr_init = config["lr_init"]
    ess_threshold = config["ess_threshold"]
    batch_size = config["batch_size"]
    refresh_batch_every = config["refresh_batch_every"]
    optim_step_start = config["optim_step_start"]
    num_particles_eval = config["num_particles_eval"]
    
    # Instantiate key iterator
    key = jax.random.PRNGKey(seed=0)
    key_iter = _get_key_iter(key)
    
    print("\n" + "="*80)
    print(f"Starting Hard 2D Gaussian Mixture Experiment with Annealing")
    print(f"mean_scale: {mean_scale}, dim: {dim}, beta_start: {beta_start}")
    print("="*80)
    
    #----------------------------------------------------------------------
    # Setup core components
    #----------------------------------------------------------------------
    # Instantiate target distributions
    target_distribution = ChallengingTwoDimensionalMixture(
        mean_scale=mean_scale, 
        dim=dim, 
        is_target=True
    )
    
    # Create intermediate target with lower beta (easier distribution)
    target_distribution_intermediate = ChallengingTwoDimensionalMixture(
        mean_scale=mean_scale, 
        dim=dim, 
        is_target=True, 
        beta=beta_start
    )
    
    # Instantiate SDE
    scheduler = LinearScheduler(t_0=t_0, t_f=t_f, beta_0=0.001, beta_f=12.0)
    sde = SDE(scheduler, sigma=sigma, dim=dim)
    
    # Instantiate potential classes for initial beta
    log_g0_intermediate = RatioPotential(sigma=sigma, target=target_distribution_intermediate)
    uncorrected_approx_potential_intermediate = NaivelyApproximatedPotential(
        base_potential=log_g0_intermediate, 
        dim=dim, 
        nn_potential_approximator=None
    )
    
    #----------------------------------------------------------------------
    # Stage 1: Evaluate naive potential approximation
    #----------------------------------------------------------------------
    print("\n### 1. Evaluating naive potential approximation")
    # Evaluate the naive potential approximation
    naive_log_Z, naive_smc_result, key, naive_ess, naive_acc_rates = evaluate_potential(
        sde, 
        uncorrected_approx_potential_intermediate, 
        num_particles, 
        ess_threshold, 
        num_steps,
        num_mcmc_steps=0  # Set > 0 to see MCMC acceptance rates
    )
    
    # Visualize the naive potential approximation samples
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    target_samples = target_distribution.sample(subkey2, num_samples=num_particles)
    naive_samples = resampler(
        rng=subkey3, 
        samples=naive_smc_result["samples"], 
        log_weights=naive_smc_result["log_weights"]
    )["samples"]
    
    # Create combined visualization
    plot_combined_visualizations(
        target_samples,
        naive_samples,
        label="Naive Approximation",
        beta=beta_start,
        ess_values=naive_ess,
        acceptance_rates=naive_acc_rates,
        mean_scale=mean_scale,
        save_filename_prefix=f"figures/PDDS/2D-hard-GM_annealing/naive_beta={beta_start:.1f}"
    )
    
    #----------------------------------------------------------------------
    # Stage 2: Train neural network potential approximation
    #----------------------------------------------------------------------
    print("\n### 2. Training neural network potential approximation")
    # Create neural network potential approximator
    nn_potential_approximator_intermediate = create_nn_potential_approximator(
        dim, 
        net_width, 
        uncorrected_approx_potential_intermediate
    )
    
    # Create loss function, optimizer and update step
    loss_fn = create_loss_fn(sde, nn_potential_approximator_intermediate, log_g0_intermediate)
    optimizer, learning_rate_schedule = create_optimizer(lr_init, lr_transition_step)
    update_step = create_update_step(loss_fn, optimizer)
    
    # Initialize the model
    initial_samples = sde.reference_dist.sample(jax.random.PRNGKey(seed=0), num_samples=batch_size)
    training_state = init_model(nn_potential_approximator_intermediate, optimizer, initial_samples, jax.random.PRNGKey(seed=0))
    
    # Log number of trainable parameters
    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
    print(f"Number of parameters: {nb_params}")
    
    # Train the network
    smc_problem = SMCProblem(sde, uncorrected_approx_potential_intermediate, num_steps)
    training_state = train_model(
        training_state, 
        update_step, 
        smc_problem, 
        batch_size, 
        refresh_batch_every, 
        optim_step_start, 
        learning_rate_schedule, 
        dim
    )
    
    # Evaluate the neural network potential approximation at initial beta
    corrected_approx_potential = NNApproximatedPotential(
        base_potential=log_g0_intermediate,
        dim=dim,
        nn_potential_approximator=partial(
            nn_potential_approximator_intermediate.apply,
            params=training_state.params_ema
        )
    )
    
    pdds_log_Z, pdds_smc_result, key, pdds_ess, pdds_acc_rates = evaluate_potential(
        sde, 
        corrected_approx_potential, 
        num_particles_eval, 
        ess_threshold, 
        num_steps,
        num_mcmc_steps=5  # Set > 0 to see MCMC acceptance rates
    )
    
    # Visualize the neural network potential approximation samples
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    target_samples = target_distribution.sample(subkey2, num_samples=num_particles_eval)
    nn_samples = resampler(
        rng=subkey3, 
        samples=pdds_smc_result["samples"], 
        log_weights=pdds_smc_result["log_weights"]
    )["samples"]
    
    # Create combined visualization
    plot_combined_visualizations(
        target_samples,
        nn_samples,
        label="PDDS",
        beta=beta_start,
        ess_values=pdds_ess,
        acceptance_rates=pdds_acc_rates,
        mean_scale=mean_scale,
        save_filename_prefix=f"figures/PDDS/2D-hard-GM_annealing/pdds_beta={beta_start:.1f}"
    )
    
    #----------------------------------------------------------------------
    # Stage 3: Continued training with annealing (increasing beta)
    #----------------------------------------------------------------------
    print("\n### 3. Continued training with annealing (increasing beta)")
    # Extract continued training parameters
    batch_size_continued = config["batch_size_continued"]
    optim_step_continued_total = config["optim_step_continued_total"] 
    num_start_repeats = config["num_start_repeats"]
    
    # Create beta schedule for annealing
    # Start with repeating initial beta, then gradually increase
    beta_values = np.linspace(beta_start+0.1, 1.0, int(round((1.0 - (beta_start+0.1))/0.1)) + 1)
    beta_list = list(np.repeat(beta_values, 2))
    optim_step_continued = optim_step_continued_total // len(beta_list)
    beta_list_valid = list(beta_values[:-1])  
    next_valid_idx = 0

    print(f"Beta schedule: {[round(b, 1) for b in beta_list]}")
    print(f"Will evaluate at beta values: {[round(b, 1) for b in beta_list_valid]}")
    
    # Dictionary to store results at each beta
    results = {
        "beta": [],
        "log_Z": [],
        "training_time": []
    }
    
    for beta_idx, beta in enumerate(beta_list):
        print(f"\nTraining iteration {beta_idx + 1}/{len(beta_list)}, beta={beta:.2f}")
        
        # Determine number of optimization steps
        if beta_idx < num_start_repeats:
            optim_step = optim_step_start
        else:
            optim_step = optim_step_continued
            
        # Update target distribution for current beta
        target_distribution_intermediate = ChallengingTwoDimensionalMixture(
            mean_scale=mean_scale, 
            dim=dim, 
            is_target=True, 
            beta=beta
        )
        log_g0 = RatioPotential(sigma=sigma, target=target_distribution_intermediate)
        uncorrected_approx_potential = NaivelyApproximatedPotential(
            base_potential=log_g0, 
            dim=dim, 
            nn_potential_approximator=None
        )
        
        # Create new neural network potential approximator
        nn_potential_approximator = create_nn_potential_approximator(
            dim,
            net_width,
            uncorrected_approx_potential
        )
        
        # Create snapshot of current parameters for generating training samples
        use_ema_params = True
        snapshot_params = training_state.params_ema if use_ema_params else training_state.params
        
        # Create and initialize potential with fixed parameters
        snapshot_approximator = create_snapshot_potential_approximator(
            nn_potential_approximator, 
            snapshot_params
        )
        
        updated_potential = NNApproximatedPotential(
            base_potential=log_g0,
            dim=dim,
            nn_potential_approximator=snapshot_approximator
        )
        
        # Create loss function using the correct nn_potential_approximator
        loss_fn = create_loss_fn(sde, nn_potential_approximator, log_g0)
        optimizer, learning_rate_schedule = create_optimizer(lr_init, lr_transition_step)
        update_step = create_update_step(loss_fn, optimizer)
        
        # Reset optimizer with current parameters
        initial_opt_state = optimizer.init(snapshot_params)
        training_state = TrainingState(
            params=snapshot_params,
            params_ema=snapshot_params,
            opt_state=initial_opt_state,
            key=jax.random.PRNGKey(seed=0),
            step=0,
        )
        
        # Create SMC problem for training
        smc_problem = SMCProblem(sde, updated_potential, num_steps)
        
        # Record training start time
        train_start_time = time.time()
        
        # Train model
        training_state = train_model(
            training_state,
            update_step,
            smc_problem,
            batch_size_continued,
            refresh_batch_every,
            optim_step,
            learning_rate_schedule,
            dim
        )
        
        # Record training end time
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        
        # Store results
        results["beta"].append(beta)
        results["training_time"].append(training_time)
        
        # Evaluate and visualize at specific beta checkpoints or at the end
        should_evaluate = ((next_valid_idx < len(beta_list_valid) and 
                           beta >= beta_list_valid[next_valid_idx] - 1e-10) or 
                           (beta_idx == len(beta_list) - 1))
        
        if should_evaluate:
            # Create final potential with trained parameters
            corrected_approx_potential = NNApproximatedPotential(
                base_potential=log_g0,
                dim=dim,
                nn_potential_approximator=partial(
                    nn_potential_approximator.apply,
                    params=training_state.params_ema
                )
            )
            
            # Determine number of samples for log Z estimation
            n_sample_for_z = 100 if beta_idx == len(beta_list) - 1 else 10
            
            # Evaluate potential
            log_Z_val, smc_result_val, key, val_ess, val_acc_rates = evaluate_potential(
                sde,
                corrected_approx_potential,
                num_particles,
                ess_threshold,
                num_steps,
                n_sample_for_z,
                num_mcmc_steps=5  # Set > 0 to see MCMC acceptance rates
            )
            
            # Store log_Z result
            results["log_Z"].append(log_Z_val)
            
            if beta_idx == len(beta_list) - 1:
                print(f'Final PDDS log Z estimate: {log_Z_val:.4f}')
                
            # Visualize samples
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            target_samples = target_distribution.sample(subkey2, num_samples=num_particles)
            annealed_samples = resampler(
                rng=subkey3,
                samples=smc_result_val["samples"],
                log_weights=smc_result_val["log_weights"]
            )["samples"]
            
            # Create combined visualization
            plot_combined_visualizations(
                target_samples,
                annealed_samples,
                label=f"PDDS (β={beta:.1f})",
                beta=beta,
                ess_values=val_ess,
                acceptance_rates=val_acc_rates,
                mean_scale=mean_scale,
                save_filename_prefix=f"figures/PDDS/2D-hard-GM_annealing/pdds_beta={beta:.1f}"
            )
            
            next_valid_idx += 1
    
    #----------------------------------------------------------------------
    # Final results and analysis
    #----------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Problem: Hard 2D Gaussian Mixture (mean_scale={mean_scale})")
    print(f"Dimension: {dim}")
    print(f"Annealing: beta from {beta_start:.2f} to 1.0")
    print(f"Neural Network: {net_width} units × 2 layers ({nb_params} parameters)")
    print("-"*80)
    print("Results by beta value:")

    sorted_results = sorted(zip(results["beta"], results["training_time"], range(len(results["beta"]))), 
                           key=lambda x: (round(x[0], 2), x[2]))

    # Group by beta value
    beta_groups = {}
    for beta, training_time, idx in sorted_results:
        rounded_beta = round(beta, 2)
        if rounded_beta not in beta_groups:
            beta_groups[rounded_beta] = []
        beta_groups[rounded_beta].append((beta, training_time, idx))

    # Print results with the proper pattern
    for rounded_beta, entries in sorted(beta_groups.items()):
        # For each unique beta value, print the evaluation result and the skipped result
        # Each unique beta value should appear in two consecutive positions in the beta_list
        # But only one of them should have evaluation (log_Z) results
        
        for i, (beta, training_time, idx) in enumerate(entries):
            if i == 0:
                # First occurrence should have evaluation result
                if idx < len(results["log_Z"]):
                    print(f"  β={rounded_beta:.2f}: log Z = {results['log_Z'][idx]:.4f}, training time: {training_time:.1f}s")
                else:
                    print(f"  β={rounded_beta:.2f}: (evaluation skipped), training time: {training_time:.1f}s")
            else:
                # Second occurrence should be skipped
                print(f"  β={rounded_beta:.2f}: (evaluation skipped), training time: {training_time:.1f}s")

    print("-"*80)
    print(f"Initial naive approx (β={beta_start:.2f}): log Z = {naive_log_Z:.4f}")
    if len(results["log_Z"]) > 0:
        print(f"Final result (β=1.00): log Z = {results['log_Z'][-1]:.4f}")
        print(f"Total improvement: {results['log_Z'][-1] - naive_log_Z:.4f}")
    print("="*80)

# Execute main function
if __name__ == "__main__":
    main()

# ## 7. Summary and Conclusions

"""
Hard 2D mixture with larger mode separation (mean_scale=3)
with progressive annealing from β=0.3 to β=1.0

Training process:
First train on easier distribution (β < 1)
Then continue training toward final target (β = 1)
Create new model initialized with previous parameters

Training parameters:
Optim steps: ~30,000 in total
Discretization: 48 steps
Batch size: 2000
Parameter transfer between stages

Results:
  β=0.40: (evaluation skipped), training time: 312.5s
  β=0.40: log Z = 1.6921, training time: 297.9s
  β=0.50: (evaluation skipped), training time: 169.0s
  β=0.50: log Z = 1.3736, training time: 165.9s
  β=0.60: (evaluation skipped), training time: 171.9s
  β=0.60: log Z = 0.9226, training time: 168.3s
  β=0.70: , training time: 173.9s
  β=0.70: log Z = 0.3553, training time: 174.6s
  β=0.80: (evaluation skipped), training time: 177.5s
  β=0.80: log Z = -0.1077, training time: 175.6s
  β=0.90: (evaluation skipped), training time: 179.6s
  β=0.90: log Z = -1044.2320, training time: 176.9s
  β=1.00: (evaluation skipped), training time: 178.9s
  β=1.00: log Z = -0.6632, training time: 170.1s
--------------------------------------------------------------------------------
Initial naive approx (β=0.30): log Z = 1.3940
Final result (β=1.00): log Z = -0.6632
Total improvement: -2.0573

Conclusion:
Annealing path may not be the correct path for learning!!!
"""


