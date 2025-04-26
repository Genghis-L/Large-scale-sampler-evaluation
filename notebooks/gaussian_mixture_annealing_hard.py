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
# Set id of available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# -

# ### Explore the naive potential approximation
# First we check out the performance of PDDS algorithm with the naive potential approximation.

# +
# global variables
dim=2
mean_scale = 2.5
sigma=1.0
t_0=0.0
t_f=1.0
num_steps = 48
num_particles = 2000
beta_start = 0.3
net_width = 64
lr_transition_step = 100
lr_init = 1e-3
ess_threshold = 0.3
figtype = 2

# INSTANTIATE OBJECTS

# Instantiate key iterator
key = jax.random.PRNGKey(seed=0)
key_iter = _get_key_iter(key)

# Instantiate target
target_distribution = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True)
target_distribution_intermediate = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True, beta=beta_start)

# Instantiate SDE
scheduler = LinearScheduler(t_0=t_0, t_f=t_f, beta_0=0.001, beta_f=12.0)
sde = SDE(scheduler, sigma=sigma, dim=dim)

# Instantiate potential classes
log_g0_intermediate = RatioPotential(sigma=sigma, target=target_distribution_intermediate)
uncorrected_approx_potential_intermediate = NaivelyApproximatedPotential(base_potential=log_g0_intermediate, dim=dim, nn_potential_approximator=None)

# MCMC step size scheduler, not used in this example so just initialised to identity function for ease
mcmc_step_size_scheduler = lambda x: x

# Instantiate SMCProblem class based on the naive approximation
smc_problem = SMCProblem(sde, uncorrected_approx_potential_intermediate, num_steps)

# Fast sampler
eval_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem,
        num_particles=(num_particles),
        ess_threshold=ess_threshold,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        density_state=0,
    )
)

# Evaluate the normalising constant estimate averaging over 100 seeds
log_Z = np.zeros(100)
for i in tqdm.trange(100, disable=True):
    key, subkey = jax.random.split(key)
    smc_result, _ = eval_sampler(subkey)
    log_Z[i] = smc_result["log_normalising_constant"]

print('Naive log Z estimate: ', np.mean(log_Z))

def plot_samples(target_samples, final_samples, label, figtype, title=None, save_filename=None):
    # Visualise target, reference and samples
    if figtype == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # Left subplot: KDE plot (as you already had)
        sns.kdeplot(target_samples[:, 0], ax=ax1, bw_adjust=0.5, label="Target")
        sns.kdeplot(final_samples[idx, 0], ax=ax1, bw_adjust=0.5, label=label)
        ax1.legend()
        ax1.set_title(f"1D Marginal Distribution")

        # Right subplot: Scatter plot of first two dimensions
        ax2.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
        ax2.scatter(final_samples[idx, 0], final_samples[idx, 1], alpha=0.5, label=label, s=10)
        ax2.legend()
        ax2.set_title(f"2D Sample Distribution")
        ax2.set_aspect('equal')  # Makes the plot have equal scale on both axes
        plt.tight_layout()

    if figtype == 2:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # Right subplot: Scatter plot of first two dimensions
        ax.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.5, label="Target", s=10)
        ax.scatter(final_samples[idx, 0], final_samples[idx, 1], alpha=0.5, label=label, s=10)
        ax.legend()
        ax.set_title(f"2D Sample Distribution")
        ax.set_aspect('equal')  # Makes the plot have equal scale on both axes
        plt.tight_layout()

    if title is not None:
    # Add an overall title
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_filename is not None:
    # Save figure with filename
        dir = os.path.dirname(save_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_filename}")

    plt.legend()
    plt.show()
    plt.close(fig)

# Visualise target, reference and samples
n_plot_samples = int(num_particles)
idx = jnp.arange(int(num_particles))
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
final_samples = resampler(
    rng=subkey3, samples=smc_result["samples"], log_weights=smc_result["log_weights"]
)["samples"]
plot_samples(target_samples, final_samples, label="Naive Approximation", figtype=figtype)
# -

# ### Learn neural network potential appproximation
# Now we introduce the neural network approximation to the log-potential function.

# Firstly we instantiate the neural network, approximate potential functions, loss functions, optimizers etc.

# +
batch_size = 600
refresh_batch_every = 100
optim_step_start = 4000

# Instantiate neural network potential approximator
@hk.without_apply_rng
@hk.transform
@check_shapes("lbd: [b]", "x: [b, d]")
def nn_potential_approximator_intermediate(lbd: Array, x: Array, density_state: int):
    std_trick = False
    std = None
    residual, density_state = uncorrected_approx_potential_intermediate.approx_log_gt(
        lbd=lbd, x=x, density_state=density_state
    )

    net = PISGRADNet(hidden_shapes=[net_width, net_width], act='gelu', dim=dim)

    out = net(lbd, x, residual)

    if std_trick:
        out = out / (std + 1e-3)

    return out, density_state

@jax.jit
@check_shapes("lbd: [b]", "x: [b, d]")
def grad_log_g(params, lbd: Array, x: Array, density_state: int):
    return x_gradient_stateful_parametrised(nn_potential_approximator_intermediate.apply)(
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
        log_g0_intermediate,
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
        log_g0_intermediate,
        False
    )

# dsm_loss_fn or guidance_loss_fn
loss_fn = guidance_loss_fn

# Instantiate learning rate schedulers
learning_rate_schedule_unlooped = optax.exponential_decay(init_value=lr_init, transition_steps=lr_transition_step, decay_rate=0.95)
learning_rate_schedule = loop_schedule(
    schedule=learning_rate_schedule_unlooped, freq=10000
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

# define model update function
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

# define haiku initialisation
@check_shapes("samples: [b, d]")
def init(samples: Array, key: Key) -> TrainingState:
    key, init_rng = jax.random.split(key)
    lbd = broadcast(jnp.array(1.0), samples)
    density_state = 0
    initial_params = nn_potential_approximator_intermediate.init(
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

# initialise haiku model
initial_samples = sde.reference_dist.sample(
    jax.random.PRNGKey(seed=0), num_samples=batch_size
)
training_state = init(initial_samples, jax.random.PRNGKey(seed=0))

# log number of trainable parameters
nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
print(f"Number of parameters: {nb_params}")
# -

# Now we train the neural network for one iteration of potential approximation.

# +
# Global variables
density_state_training = 0

# Initial sampler for training samples
smc_problem = SMCProblem(sde, uncorrected_approx_potential_intermediate, num_steps)
training_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem,
        num_particles=batch_size * refresh_batch_every,
        ess_threshold=ess_threshold,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
    )
)
# initial jit compilation
_, _ = training_sampler(rng=key, density_state=0)

progress_bar = tqdm.tqdm(
    list(range(1, optim_step_start + 1)),
    miniters=1,
    disable=False,
)

start_time = time.time()
for step, key in zip(progress_bar, key_iter):
    # Generate samples for potential approximation training from PDDS with naive potential approximation
    if (
        step - 1
    ) % refresh_batch_every == 0:  # refresh samples after every 'epoch'
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
print('Training complete, training time: ', end_time - start_time)

# +
# initialise the new potential approximation using the learnt network
num_particles = 4000

corrected_approx_potential = NNApproximatedPotential(
    base_potential=log_g0_intermediate,
    dim=dim,
    nn_potential_approximator=partial(
        nn_potential_approximator_intermediate.apply,
        params=training_state.params_ema
    )
)

# Instantiate SMCProblem class based on the learnt potential approximation
smc_problem = SMCProblem(sde, corrected_approx_potential, num_steps)

# Fast sampler
eval_sampler = jax.jit(
    partial(
        fast_outer_loop_smc,
        smc_problem=smc_problem,
        num_particles=(num_particles),
        ess_threshold=ess_threshold,
        num_mcmc_steps=0,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        density_state=0,
    )
)

# Evaluate the normalising constant estimate averaging over 100 seeds
log_Z = np.zeros(100)
for i in tqdm.trange(100, disable=True):
    key, subkey = jax.random.split(key)
    smc_result, _ = eval_sampler(subkey)
    log_Z[i] = smc_result["log_normalising_constant"]
print('PDDS log Z estimate: ', np.mean(log_Z))

# Visualise samples
n_plot_samples = int(num_particles)
idx = jnp.arange(int(num_particles))
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
final_samples = resampler(
    rng=subkey3, samples=smc_result["samples"], log_weights=smc_result["log_weights"]
)["samples"]

plot_samples(
    target_samples, final_samples, label="PDDS", figtype=figtype,
    title=f"β={beta_start:.3f}",
    save_filename=f"tmp_figs/mixture_comparison_beta_{beta_start:.3f}.png"
)
# -

# # Continued training

# optim_step = 4000
# for beta in np.arange(0.3, 0.5, 0.025):
batch_size = 1000
refresh_batch_every = 100
optim_step_continued_total = 30000
num_start_repeats = 2
beta_list = [beta_start] * num_start_repeats + list(np.repeat(np.arange(beta_start+0.1, 1.001, 0.1), 2))
optim_step_continued = optim_step_continued_total // (len(beta_list) - num_start_repeats)
beta_list_valid = list(np.arange(beta_start+0.1, 0.99, 0.1))
next_valid_idx = 0
for beta_idx, beta in enumerate(beta_list):
    if beta_idx < num_start_repeats:
        optim_step = optim_step_start
    else:
        optim_step = optim_step_continued
    target_distribution_intermediate = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True, beta=beta)
    log_g0 = RatioPotential(sigma=sigma, target=target_distribution_intermediate)
    uncorrected_approx_potential = NaivelyApproximatedPotential(base_potential=log_g0, dim=dim, nn_potential_approximator=None)

    # Instantiate neural network potential approximator
    @hk.without_apply_rng
    @hk.transform
    @check_shapes("lbd: [b]", "x: [b, d]")
    def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
        std_trick = False
        std = None
        residual, density_state = uncorrected_approx_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )

        net = PISGRADNet(hidden_shapes=[net_width, net_width], act='gelu', dim=dim)

        out = net(lbd, x, residual)

        if std_trick:
            out = out / (std + 1e-3)

        return out, density_state

    def create_snapshot_potential_approximator(params):
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

    # Create a potential object B with the current snapshot of parameters
    # You can choose either regular params or EMA params
    use_ema_params = True  # Set to False if you want to use regular params
    snapshot_params = training_state.params_ema if use_ema_params else training_state.params
    initial_opt_state = optimizer.init(snapshot_params)
    TrainingState(
        params=snapshot_params,
        params_ema=training_state.params_ema,
        opt_state=initial_opt_state,
        key=jax.random.PRNGKey(seed=0),
        step=0,
    )

    # Create the approximator with fixed snapshot parameters
    snapshot_approximator = create_snapshot_potential_approximator(snapshot_params)

    # Create the NNApproximatedPotential with the snapshot
    updated_potential = NNApproximatedPotential(
        base_potential=log_g0,
        dim=dim,  # Use the same dimension as in your training
        nn_potential_approximator=snapshot_approximator
    )

    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_potential_approximator_intermediate.apply)(
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

    # dsm_loss_fn or guidance_loss_fn
    loss_fn = guidance_loss_fn

    # Instantiate learning rate schedulers
    learning_rate_schedule_unlooped = optax.exponential_decay(init_value=lr_init, transition_steps=lr_transition_step, decay_rate=0.95)
    learning_rate_schedule = loop_schedule(
        schedule=learning_rate_schedule_unlooped, freq=10000
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    # define model update function
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

    # Global variables
    density_state_training = 0

    # Initial sampler for training samples
    smc_problem = SMCProblem(sde, updated_potential, num_steps)
    training_sampler = jax.jit(
        partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=batch_size * refresh_batch_every,
            ess_threshold=ess_threshold,
            num_mcmc_steps=0,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        )
    )

    # initial jit compilation
    _, _ = training_sampler(rng=key, density_state=0)

    progress_bar = tqdm.tqdm(
        list(range(1, optim_step + 1)),
        miniters=1,
        disable=False,
    )

    start_time = time.time()
    for step, key in zip(progress_bar, key_iter):
        # Generate samples for potential approximation training from PDDS with naive potential approximation
        if (
            step - 1
        ) % refresh_batch_every == 0:  # refresh samples after every 'epoch'
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
    print('Training complete, training time: ', end_time - start_time)

    # initialise the new potential approximation using the learnt network
    num_particles = 2000
    corrected_approx_potential = NNApproximatedPotential(
        base_potential=log_g0,
        dim=dim,
        nn_potential_approximator=partial(
            nn_potential_approximator.apply,
            params=training_state.params_ema
        )
    )

    if (next_valid_idx < len(beta_list_valid) and beta >= beta_list_valid[next_valid_idx] - 1e-10) or (beta_idx == len(beta_list) - 1):
        # Trigger the validation sample and plot
        # Instantiate SMCProblem class based on the learnt potential approximation
        smc_problem = SMCProblem(sde, corrected_approx_potential, num_steps)

        # Fast sampler
        eval_sampler = jax.jit(
            partial(
                fast_outer_loop_smc,
                smc_problem=smc_problem,
                num_particles=(num_particles),
                ess_threshold=ess_threshold,
                num_mcmc_steps=0,
                mcmc_step_size_scheduler=mcmc_step_size_scheduler,
                density_state=0,
            )
        )

        # Evaluate the normalising constant estimate averaging over 100 seeds at the end; otherwise, only sample once
        if beta_idx == len(beta_list) - 1:
            n_sample_for_z = 100
        else:
            n_sample_for_z = 1
        log_Z = np.zeros(n_sample_for_z)
        for i in tqdm.trange(n_sample_for_z, disable=True):
            key, subkey = jax.random.split(key)
            smc_result, _ = eval_sampler(subkey)
            log_Z[i] = smc_result["log_normalising_constant"]
        if beta_idx == len(beta_list) - 1:
            print('PDDS log Z estimate: ', np.mean(log_Z))

        # Visualise samples
        n_plot_samples = int(num_particles)
        idx = jnp.arange(int(num_particles))
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
        final_samples = resampler(
            rng=subkey3, samples=smc_result["samples"], log_weights=smc_result["log_weights"]
        )["samples"]

        plot_samples(
            target_samples, final_samples, label="PDDS", figtype=figtype,
            title=f"β={beta:.3f}",
            save_filename=f"tmp_figs/mixture_comparison_beta_{beta:.3f}.png"
        )

        next_valid_idx += 1


