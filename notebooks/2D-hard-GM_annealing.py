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

# # Gaussian Mixture Interactive Example

# ### Setup

# +
# # %load_ext autoreload
# # %autoreload 2

import os
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

from pdds.sde import SDE, guidance_loss, dsm_loss
from pdds.smc_problem import SMCProblem
from pdds.potentials import NNApproximatedPotential
from pdds.utils.shaping import broadcast
from pdds.utils.jax import (
    _get_key_iter,
    x_gradient_stateful_parametrised,
)
from pdds.utils.lr_schedules import loop_schedule
from pdds.ml_tools.state import TrainingState
from pdds.smc_loops import fast_outer_loop_smc
from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture
from pdds.sde import LinearScheduler, SDE
from pdds.potentials import RatioPotential, NaivelyApproximatedPotential, NNApproximatedPotential
from pdds.nn_models.mlp import PISGRADNet
from pdds.resampling import resampler
from pdds.distributions import NormalDistribution

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

# ### Configuration

def get_config():
    """Return configuration parameters for the experiment"""
    config = {
        # Environment settings
        "cuda_visible_devices": "0",
        
        # Problem dimension settings
        "dim": 2,
        "mean_scale": 3.0,
        "sigma": 1.0,
        
        # SDE settings
        "t_0": 0.0,
        "t_f": 1.0,
        "num_steps": 48,
        
        # SMC settings
        "num_particles": 2000,
        "ess_threshold": 0.3,
        
        # Training settings
        "beta_start": 0.3,
        "net_width": 64,
        "lr_transition_step": 100,
        "lr_init": 1e-3,
        "batch_size": 600,
        "refresh_batch_every": 100,
        "optim_step_start": 4000,
        
        # Continued training settings
        "batch_size_continued": 1000,
        "optim_step_continued_total": 30000,
        "num_start_repeats": 2,
        
        # Evaluation settings
        "n_eval_samples": 100,
        "num_particles_eval": 4000,
        
        # Visualization
        "figtype": 2,
    }
    return config

# Set configuration
config = get_config()

# Set id of available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = config["cuda_visible_devices"]

# ### Visualization functions

def plot_2d_scatter(target_samples, final_samples, label, title=None, save_filename=None, mean_scale=None):
    """Plot scatter comparison between target and sampled distributions."""
    plt.figure(figsize=(8, 6))
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
    plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, label=label, s=10)
    
    if title:
        plt.title(title)
    elif mean_scale is not None:
        plt.title(f"Hard 2D Gaussian Mixture (mean_scale={mean_scale})")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_filename:
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

def plot_ess_over_time(ess_values, title=None, save_filename=None):
    """Plot effective sample size across timesteps.
    
    Args:
        ess_values: Array of ESS values at each timestep
        title: Plot title 
        save_filename: File to save plot to
    """
    plt.figure(figsize=(10, 5))
    steps = np.arange(len(ess_values))
    plt.plot(steps, ess_values, marker='o', markersize=4, linestyle='-')
    plt.xlabel('Time step')
    plt.ylabel('Effective Sample Size (ESS)')
    plt.grid(True)
    
    if title:
        plt.title(title)
    else:
        plt.title('Effective Sample Size at Each Time Step')
    
    plt.tight_layout()
    
    if save_filename:
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

def plot_mcmc_acceptance_rates(acceptance_rates, title=None, save_filename=None):
    """Plot MCMC acceptance rates across timesteps.
    
    Args:
        acceptance_rates: Array of acceptance rates at each timestep
        title: Plot title
        save_filename: File to save plot to
    """
    if acceptance_rates is None or len(acceptance_rates) == 0:
        print("No MCMC acceptance rates to plot (MCMC steps may be disabled)")
        return
        
    plt.figure(figsize=(10, 5))
    steps = np.arange(len(acceptance_rates))
    plt.plot(steps, acceptance_rates, marker='o', markersize=4, linestyle='-')
    plt.xlabel('Time step')
    plt.ylabel('MCMC Acceptance Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    
    if title:
        plt.title(title)
    else:
        plt.title('MCMC Acceptance Rate at Each Time Step')
    
    plt.tight_layout()
    
    if save_filename:
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

def plot_1d_marginals(target_samples, sampled_samples, label, beta=None, save_filename_prefix=None):
    """Plot 1D marginal distributions for x and y axes.
    
    Args:
        target_samples: Samples from target distribution
        sampled_samples: Samples from approximation
        label: Label for approximation
        beta: Beta value for title
        save_filename_prefix: Prefix for saving plots
    """
    # X-axis marginal
    plt.figure(figsize=(10, 5))
    sns.kdeplot(target_samples[:, 0], label="Target", alpha=0.7)
    sns.kdeplot(sampled_samples[:, 0], label=label, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("Density")
    beta_str = f" (β={beta:.1f})" if beta is not None else ""
    plt.title(f"1D Marginal Distribution (x-axis){beta_str}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_filename_prefix:
        save_filename = f"{save_filename_prefix}_marginal_x.png"
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()
    
    # Y-axis marginal
    plt.figure(figsize=(10, 5))
    sns.kdeplot(target_samples[:, 1], label="Target", alpha=0.7)
    sns.kdeplot(sampled_samples[:, 1], label=label, alpha=0.7)
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title(f"1D Marginal Distribution (y-axis){beta_str}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_filename_prefix:
        save_filename = f"{save_filename_prefix}_marginal_y.png"
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

def plot_combined_visualizations(target_samples, final_samples, label, beta=None, ess_values=None, 
                               acceptance_rates=None, mean_scale=None, save_filename_prefix=None):
    """Create a combined visualization with 1D marginals and 2D scatter in one row,
    and ESS/acceptance rates in a second row if available.
    
    Args:
        target_samples: Samples from target distribution
        final_samples: Samples from approximation
        label: Label for approximation
        beta: Beta value for title
        ess_values: ESS values over time
        acceptance_rates: MCMC acceptance rates
        mean_scale: Mean scale for title
        save_filename_prefix: Prefix for saving plots
    """
    beta_str = f" (β={beta:.1f})" if beta is not None else ""
    
    # Create figure with appropriate layout
    has_monitoring = (ess_values is not None or acceptance_rates is not None)
    
    if has_monitoring:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3)
    else:
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3)
    
    # First row: 1D x-marginal, 1D y-marginal, 2D scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Second row (if needed): ESS and acceptance rate
    if has_monitoring:
        if ess_values is not None:
            ax4 = fig.add_subplot(gs[1, 0:2])
        if acceptance_rates is not None:
            ax5 = fig.add_subplot(gs[1, 2])
    
    # Plot 1D Marginal X
    sns.kdeplot(target_samples[:, 0], label="Target", alpha=0.7, ax=ax1)
    sns.kdeplot(final_samples[:, 0], label=label, alpha=0.7, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.set_title(f"1D Marginal Distribution (x-axis){beta_str}")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 1D Marginal Y
    sns.kdeplot(target_samples[:, 1], label="Target", alpha=0.7, ax=ax2)
    sns.kdeplot(final_samples[:, 1], label=label, alpha=0.7, ax=ax2)
    ax2.set_xlabel("y")
    ax2.set_ylabel("Density")
    ax2.set_title(f"1D Marginal Distribution (y-axis){beta_str}")
    ax2.grid(True)
    ax2.legend()
    
    # Plot 2D Scatter
    ax3.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
    ax3.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, label=label, s=10)
    title = f"Hard 2D Gaussian Mixture {beta_str}"
    if mean_scale is not None:
        title += f" (mean_scale={mean_scale})"
    ax3.set_title(title)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Plot ESS values if available
    if ess_values is not None:
        steps = np.arange(len(ess_values))
        ax4.plot(steps, ess_values, marker='o', markersize=4, linestyle='-')
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Effective Sample Size (ESS)')
        ax4.set_title(f"ESS Values {beta_str}")
        ax4.grid(True)
        
        # Add annotation explaining why ESS might be missing if it's very sparse
        if len(ess_values) <= 1:
            ax4.annotate(
                "Note: ESS history may be missing because:\n"
                "1. fast_outer_loop_smc may not be storing ESS history\n"
                "2. Try adding mcmc_steps>0 to see acceptance rates",
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3)
            )
    
    # Plot MCMC acceptance rates if available
    if acceptance_rates is not None:
        steps = np.arange(len(acceptance_rates))
        ax5.plot(steps, acceptance_rates, marker='o', markersize=4, linestyle='-')
        ax5.set_xlabel('Time step')
        ax5.set_ylabel('MCMC Acceptance Rate')
        ax5.set_ylim(0, 1)
        ax5.set_title(f"MCMC Acceptance Rates {beta_str}")
        ax5.grid(True)
        
        # Add annotation explaining acceptance rate might be missing
        if len(acceptance_rates) <= 1:
            ax5.annotate(
                "Note: Acceptance rates are missing because\n"
                "num_mcmc_steps=0. Set num_mcmc_steps>0\n"
                "in evaluate_potential() to see data.",
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3)
            )
    
    plt.tight_layout()
    
    if save_filename_prefix:
        save_filename = f"{save_filename_prefix}_combined.png"
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")
    
    plt.show()
    plt.close()

# ### Model training helpers

def create_nn_potential_approximator(dim, net_width, base_potential):
    """Create a neural network potential approximator."""
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
    """Create loss function for training the potential approximator."""
    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_potential_approximator.apply)(
            params, lbd, x, density_state
        )

    # Define loss function
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
    """Create optimizer with learning rate schedule."""
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

def create_update_step(loss_fn, optimizer):
    """Create function for updating model parameters."""
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
            lambda p_ema, p: p_ema * 0.999
            + p * (1.0 - 0.999),
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
    """Initialize the neural network model."""
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

def evaluate_potential(sde, potential, num_particles, ess_threshold, num_steps, n_samples=100, num_mcmc_steps=0):
    """Evaluate a potential by estimating the log normalizing constant.
    
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
    
    for i in tqdm.trange(n_samples, disable=True):
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
    
    return np.mean(log_Z), samples, key, ess_values, acceptance_rates

def create_snapshot_potential_approximator(nn_potential_approximator, params):
    """Create a potential approximator with fixed parameters."""
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
    """Train the neural network model."""
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
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")
            
    end_time = time.time()
    print(f'Training complete, training time: {end_time - start_time:.2f}s')
    
    return training_state

# ### Main experiment

def main():
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
    
    # Instantiate target distributions
    target_distribution = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True)
    target_distribution_intermediate = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True, beta=beta_start)
    
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
    
    # MCMC step size scheduler (identity function for this example)
    mcmc_step_size_scheduler = lambda x: x
    
    print("\n### 1. Evaluating naive potential approximation")
    # Evaluate the naive potential approximation
    naive_log_Z, naive_smc_result, key, naive_ess, naive_acc_rates = evaluate_potential(
        sde, 
        uncorrected_approx_potential_intermediate, 
        num_particles, 
        ess_threshold, 
        num_steps,
        num_mcmc_steps=0  # 需要设置大于0的值才能看到MCMC接受率
    )
    print(f'Naive log Z estimate: {naive_log_Z:.4f}')
    
    # Visualize the naive potential approximation samples
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    target_samples = target_distribution.sample(subkey2, num_samples=num_particles)
    final_samples = resampler(
        rng=subkey3, 
        samples=naive_smc_result["samples"], 
        log_weights=naive_smc_result["log_weights"]
    )["samples"]
    
    # Create combined visualization
    plot_combined_visualizations(
        target_samples,
        final_samples,
        label="Naive Approximation",
        beta=beta_start,
        ess_values=naive_ess,
        acceptance_rates=naive_acc_rates,
        mean_scale=mean_scale,
        save_filename_prefix=f"figures/naive_beta={beta_start:.1f}"
    )
    
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
    
    # Evaluate the neural network potential approximation
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
        num_mcmc_steps=5  # Need to set greater than 0 to see MCMC acceptance rates
    )
    print(f'PDDS log Z estimate: {pdds_log_Z:.4f}')
    
    # Visualize the neural network potential approximation samples
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    target_samples = target_distribution.sample(subkey2, num_samples=num_particles_eval)
    final_samples = resampler(
        rng=subkey3, 
        samples=pdds_smc_result["samples"], 
        log_weights=pdds_smc_result["log_weights"]
    )["samples"]
    
    # Create combined visualization
    plot_combined_visualizations(
        target_samples,
        final_samples,
        label="PDDS",
        beta=beta_start,
        ess_values=pdds_ess,
        acceptance_rates=pdds_acc_rates,
        mean_scale=mean_scale,
        save_filename_prefix=f"figures/pdds_beta={beta_start:.1f}"
    )
    
    print("\n### 3. Continued training with increasing beta")
    # Extract continued training parameters
    batch_size_continued = config["batch_size_continued"]
    optim_step_continued_total = config["optim_step_continued_total"] 
    num_start_repeats = config["num_start_repeats"]
    
    # Create beta schedule
    beta_list = [beta_start] * num_start_repeats + list(np.repeat(np.arange(beta_start+0.1, 1.001, 0.1), 2))
    optim_step_continued = optim_step_continued_total // (len(beta_list) - num_start_repeats)
    beta_list_valid = list(np.arange(beta_start+0.1, 0.99, 0.1))
    next_valid_idx = 0
    
    for beta_idx, beta in enumerate(beta_list):
        print(f"\nTraining iteration {beta_idx + 1}/{len(beta_list)}, beta={beta:.2f}")
        
        # Determine number of optimization steps
        if beta_idx < num_start_repeats:
            optim_step = optim_step_start
        else:
            optim_step = optim_step_continued
            
        # Update target distribution and potential
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
        
        # Create loss function using the CORRECT nn_potential_approximator
        # (This fixes the bug in the original code that used nn_potential_approximator_intermediate)
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
        
        # Evaluate and visualize at specific beta values or at the end
        if (next_valid_idx < len(beta_list_valid) and beta >= beta_list_valid[next_valid_idx] - 1e-10) or (beta_idx == len(beta_list) - 1):
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
            n_sample_for_z = 100 if beta_idx == len(beta_list) - 1 else 1
            
            # Evaluate potential
            log_Z_val, smc_result_val, key, val_ess, val_acc_rates = evaluate_potential(
                sde,
                corrected_approx_potential,
                num_particles,
                ess_threshold,
                num_steps,
                n_sample_for_z,
                num_mcmc_steps=0  # 需要设置大于0的值才能看到MCMC接受率
            )
            
            if beta_idx == len(beta_list) - 1:
                print(f'Final PDDS log Z estimate: {log_Z_val:.4f}')
                
            # Visualize samples
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            target_samples = target_distribution.sample(subkey2, num_samples=num_particles)
            final_samples = resampler(
                rng=subkey3,
                samples=smc_result_val["samples"],
                log_weights=smc_result_val["log_weights"]
            )["samples"]
            
            # Create combined visualization
            plot_combined_visualizations(
                target_samples,
                final_samples,
                label="PDDS",
                beta=beta,
                ess_values=val_ess,
                acceptance_rates=val_acc_rates,
                mean_scale=mean_scale,
                save_filename_prefix=f"figures/pdds_beta={beta:.1f}"
            )
            
            next_valid_idx += 1

# Execute main function
if __name__ == "__main__":
    main()


