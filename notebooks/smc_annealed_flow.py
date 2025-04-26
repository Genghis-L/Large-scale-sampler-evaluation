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

# # Sequential Monte Carlo (SMC) with Annealed Flow Transport

# ### Setup

# +
# %load_ext autoreload
# %autoreload 2

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the appropriate GPU ID
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
from typing import Tuple, Dict, Any, Callable

from jaxtyping import PRNGKeyArray as Key, Array
import typing as tp
from check_shapes import check_shapes

from pdds.sde import SDE, guidance_loss, dsm_loss
from pdds.smc_problem import SMCProblem
from pdds.potentials import NNApproximatedPotential, NaivelyApproximatedPotential, RatioPotential
from pdds.utils.shaping import broadcast
from pdds.utils.jax import _get_key_iter, x_gradient_stateful_parametrised
from pdds.ml_tools.state import TrainingState
from pdds.smc_loops import fast_outer_loop_smc
from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture
from pdds.sde import LinearScheduler, SDE
from pdds.resampling import optionally_resample, essl, resampler

# -

# ### Implement SMC methods from annealed flow transport

# +
class AnnealedFlowSMC:
    """Implementation of Sequential Monte Carlo (SMC) sampler algorithm from annealed_flow_transport."""
    
    def __init__(self, smc_problem: SMCProblem, config: Dict):
        """Initialize SMC sampler.
        
        Args:
            smc_problem: SMC problem object
            config: Configuration dictionary with parameters:
                - num_particles: Number of particles
                - num_steps: Number of annealing steps
                - use_resampling: Whether to use resampling
                - ess_threshold: Threshold for effective sample size ratio
                - num_mcmc_steps: Number of MCMC steps
                - mcmc_step_size: Step size for MCMC
                - report_step: How often to report progress
        """
        self.smc_problem = smc_problem
        self.config = config
        
        # Extract config parameters with defaults
        self.num_particles = config.get('num_particles', 1000)
        self.num_steps = config.get('num_steps', 10)
        self.use_resampling = config.get('use_resampling', True)
        self.ess_threshold = config.get('ess_threshold', 0.3)
        self.num_mcmc_steps = config.get('num_mcmc_steps', 0)
        self.mcmc_step_size = config.get('mcmc_step_size', 0.1)
        self.report_step = config.get('report_step', 1)
        
    @check_shapes("samples: [b, d]", "log_weights: [b]")
    def inner_loop(self, 
                  rng: Key,
                  t_new: Array,
                  t_prev: Array,
                  samples: Array,
                  log_weights: Array,
                  density_state: int) -> Tuple[Dict, int]:
        """Single step of the SMC algorithm inner loop.
        
        Args:
            rng: Random key
            t_new: New time step
            t_prev: Previous time step
            samples: Samples from previous time step
            log_weights: Log weights from previous time step
            density_state: State of the density estimator
            
        Returns:
            Dictionary with:
                - samples_new: Samples after update
                - log_weights_new: Log weights after update
                - log_normaliser_increment: Log normalizer increment
                - acceptance_ratio: Acceptance ratio
                - resampled: Whether resampling was performed
            Updated density state
        """
        # Move - generate proposal
        rng, rng_ = jax.random.split(rng)
        proposal, density_state = self.smc_problem.markov_kernel_apply(
            x_prev=samples, t_new=t_new, t_prev=t_prev, density_state=density_state
        )
        samples_just_before_resampling = proposal.sample(rng_, self.num_particles)
        
        # Reweight
        lw_incr, density_state = self.smc_problem.reweighter(
            x_new=samples_just_before_resampling,
            x_prev=samples,
            t_new=t_new,
            t_prev=t_prev,
            density_state=density_state,
        )
        log_weights_just_before_resampling = jax.nn.log_softmax(log_weights + lw_incr)
        log_normaliser_increment = jnp.sum(jax.nn.softmax(log_weights) * lw_incr)
        
        # Resample if needed
        if self.use_resampling:
            rng, rng_ = jax.random.split(rng)
            resample_result = optionally_resample(
                rng=rng_,
                log_weights=log_weights_just_before_resampling,
                samples=samples_just_before_resampling,
                ess_threshold=self.ess_threshold
            )
            resampled_samples = resample_result["samples"]
            log_weights_resampled = resample_result["lw"]
            resampled = resample_result["resampled"]
        else:
            resampled_samples = samples_just_before_resampling
            log_weights_resampled = log_weights_just_before_resampling
            resampled = False
            
        # Apply MCMC steps if configured
        if self.num_mcmc_steps > 0 and resampled:
            rng, rng_ = jax.random.split(rng)
            MCMC_kernel = self.smc_problem.get_MCMC_kernel(t_new, self.mcmc_step_size)
            keys = jax.random.split(rng_, self.num_mcmc_steps)
            (samples_new, density_state), acceptance_rates = jax.lax.scan(
                MCMC_kernel, (resampled_samples, density_state), keys
            )
            acceptance_ratio = jnp.mean(acceptance_rates)
        else:
            samples_new = resampled_samples
            acceptance_ratio = 1.0
            
        return {
            "samples_new": samples_new,
            "log_weights_new": log_weights_resampled,
            "log_normaliser_increment": log_normaliser_increment,
            "acceptance_ratio": acceptance_ratio,
            "resampled": resampled
        }, density_state
        
    def fast_outer_loop(self, rng: Key, density_state: int = 0) -> Dict:
        """A fast SMC loop for evaluation.
        
        Args:
            rng: Random key
            density_state: Initial density state
            
        Returns:
            Dictionary with:
                - samples: Final samples
                - log_weights: Final log weights
                - log_normalising_constant: Log normalizing constant estimate
                - ess_log: ESS at each step
                - acceptance_log: Acceptance ratio at each step
                - logZ_incr_log: Log normalizer increment at each step
        """
        rng, rng_ = jax.random.split(rng)
        
        # Initialize samples and weights
        x = self.smc_problem.initial_distribution.sample(rng_, self.num_particles)
        lw_unnorm, density_state = self.smc_problem.initial_reweighter(x, density_state)
        lw = jax.nn.log_softmax(lw_unnorm)
        logZ = jnp.sum(jax.nn.softmax(lw_unnorm) * lw_unnorm)
        initial_ess = essl(lw)
        
        # Optional initial resampling
        if self.use_resampling:
            rng, rng_ = jax.random.split(rng)
            initial_resample = optionally_resample(rng_, lw, x, self.ess_threshold)
            x = initial_resample["samples"]
            lw = initial_resample["lw"]
            
        # Setup logging arrays
        ess_log = np.zeros(self.num_steps + 1)
        acceptance_log = np.zeros(self.num_steps + 1)
        logZ_incr_log = np.zeros(self.num_steps + 1)
        
        ess = essl(lw)
        ess_log[0] = ess
        acceptance_log[0] = 1.0
        logZ_incr_log[0] = 0.0
        
        # JIT compile inner loop
        inner_loop_jit = jax.jit(self.inner_loop)
        
        # Time discretization
        ts = jnp.linspace(0.0, self.smc_problem.tf, self.num_steps + 1)
        t1 = jnp.flip(ts[:-1])
        t2 = jnp.flip(ts[1:])
        
        # Generate random keys for each step
        keys = jax.random.split(rng, self.num_steps)
        
        # Run the SMC algorithm
        for i, (t_new, t_prev) in tqdm.tqdm(enumerate(zip(t1, t2))):
            rng_ = keys[i]
            inner_loop_result, density_state = inner_loop_jit(
                rng_,
                t_new=t_new,
                t_prev=t_prev,
                samples=x,
                log_weights=lw,
                density_state=density_state
            )
            
            x = inner_loop_result["samples_new"]
            lw = inner_loop_result["log_weights_new"]
            logZ_incr_log[i + 1] = inner_loop_result["log_normaliser_increment"]
            logZ += inner_loop_result["log_normaliser_increment"]
            
            ess = essl(lw)
            ess_log[i + 1] = ess
            acceptance_log[i + 1] = inner_loop_result["acceptance_ratio"]
            
            # Report progress if configured
            if (i + 1) % self.report_step == 0:
                print(f"Step {i+1}/{self.num_steps}: ESS={ess:.2f}, Acceptance={inner_loop_result['acceptance_ratio']:.4f}")
                
        return {
            "samples": x,
            "log_weights": lw,
            "log_normalising_constant": logZ,
            "ess_log": ess_log,
            "acceptance_log": acceptance_log,
            "logZ_incr_log": logZ_incr_log,
            "initial_ess": initial_ess
        }
    
    def outer_loop(self, rng: Key, density_state: int = 0) -> Dict:
        """Full SMC loop with timing and diagnostics.
        
        Args:
            rng: Random key
            density_state: Initial density state
            
        Returns:
            Dictionary with:
                - samples: Final samples
                - log_weights: Final log weights
                - log_normalising_constant: Log normalizing constant estimate
                - ess_log: ESS at each step
                - acceptance_log: Acceptance ratio at each step
                - logZ_incr_log: Log normalizer increment at each step
                - delta_time: Total time taken
        """
        start_time = time.time()
        result = self.fast_outer_loop(rng, density_state)
        finish_time = time.time()
        delta_time = finish_time - start_time
        
        print(f"Total time: {delta_time:.2f} seconds")
        print(f"Log normaliser estimate: {result['log_normalising_constant']:.4f}")
        
        result["delta_time"] = delta_time
        return result
# -

# ### Example with Gaussian Mixture

# +
# global variables
dim = 2
mean_scale = 3.0
sigma = 1.0
t_0 = 0.0
t_f = 1.0
num_steps = 24
num_particles = 2000
beta = 0.6

# INSTANTIATE KEY ITERATOR
key = jax.random.PRNGKey(seed=0)
key_iter = _get_key_iter(key)

# INSTANTIATE TARGET DISTRIBUTIONS
target_distribution = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True)
target_distribution_intermediate = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True, beta=beta)

# INSTANTIATE SDE
scheduler = LinearScheduler(t_0=t_0, t_f=t_f, beta_0=0.001, beta_f=12.0)
sde = SDE(scheduler, sigma=sigma, dim=dim)

# INSTANTIATE POTENTIAL CLASSES
log_g0_intermediate = RatioPotential(sigma=sigma, target=target_distribution_intermediate)
uncorrected_approx_potential_intermediate = NaivelyApproximatedPotential(base_potential=log_g0_intermediate, dim=dim, nn_potential_approximator=None)

# MCMC step size scheduler (identity function for simplicity)
mcmc_step_size_scheduler = lambda x: 0.1

# Instantiate SMCProblem class based on the naive approximation
smc_problem = SMCProblem(sde, uncorrected_approx_potential_intermediate, num_steps)

# Create configuration for SMC
smc_config = {
    'num_particles': num_particles,
    'num_steps': num_steps,
    'use_resampling': True,
    'ess_threshold': 0.3,
    'num_mcmc_steps': 5,
    'mcmc_step_size': 0.1,
    'report_step': 4
}

# Create Annealed Flow SMC instance
annealed_smc = AnnealedFlowSMC(smc_problem, smc_config)

# Run the SMC algorithm
key, subkey = jax.random.split(key)
result = annealed_smc.outer_loop(subkey)

# +
# Plot ESS and acceptance rate through the algorithm
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(result['ess_log'])
plt.title('Effective Sample Size')
plt.xlabel('Step')
plt.ylabel('ESS')

plt.subplot(1, 2, 2)
plt.plot(result['acceptance_log'])
plt.title('MCMC Acceptance Rate')
plt.xlabel('Step')
plt.ylabel('Acceptance Rate')

plt.tight_layout()
plt.show()

# Visualize target and samples
n_plot_samples = int(num_particles)
key, subkey = jax.random.split(key)
target_samples = target_distribution.sample(subkey, num_samples=n_plot_samples)

# Ensure samples are properly weighted - resample if necessary
key, subkey = jax.random.split(key)
final_samples = resampler(
    rng=subkey, samples=result["samples"], log_weights=result["log_weights"]
)["samples"]

plt.figure(figsize=(15, 5))

# 1D plot
plt.subplot(1, 3, 1)
sns.kdeplot(final_samples[:, 0], label="SMC")
sns.kdeplot(target_samples[:, 0], label="Target")
plt.legend()
plt.title("1D Marginal (x-axis)")

# 1D plot - y dimension
plt.subplot(1, 3, 2)
sns.kdeplot(final_samples[:, 1], label="SMC")
sns.kdeplot(target_samples[:, 1], label="Target")
plt.legend()
plt.title("1D Marginal (y-axis)")

# 2D scatter plot
plt.subplot(1, 3, 3)
plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.3, label="SMC")
plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.3, label="Target")
plt.legend()
plt.title("2D Samples")

plt.tight_layout()
plt.show()

# +
# Compare with existing fast_outer_loop_smc implementation
key, subkey = jax.random.split(key)
pdds_result, _ = fast_outer_loop_smc(
    rng=subkey,
    smc_problem=smc_problem,
    num_particles=num_particles,
    ess_threshold=0.3,
    num_mcmc_steps=5,
    mcmc_step_size_scheduler=mcmc_step_size_scheduler,
    density_state=0
)

# Ensure samples are properly weighted - resample if necessary
key, subkey = jax.random.split(key)
pdds_final_samples = resampler(
    rng=subkey, samples=pdds_result["samples"], log_weights=pdds_result["log_weights"]
)["samples"]

# Compare log normalizing constants
print(f"Annealed Flow SMC log Z estimate: {result['log_normalising_constant']:.4f}")
print(f"PDDS SMC log Z estimate: {pdds_result['log_normalising_constant']:.4f}")

# Plot comparison of samples
plt.figure(figsize=(12, 4))

# 1D comparison
plt.subplot(1, 2, 1)
sns.kdeplot(final_samples[:, 0], label="Annealed Flow SMC")
sns.kdeplot(pdds_final_samples[:, 0], label="PDDS SMC")
sns.kdeplot(target_samples[:, 0], label="Target")
plt.legend()
plt.title("1D Comparison")

# 2D scatter
plt.subplot(1, 2, 2)
plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.3, label="Annealed Flow SMC")
plt.scatter(pdds_final_samples[:, 0], pdds_final_samples[:, 1], alpha=0.3, label="PDDS SMC")
plt.legend()
plt.title("2D Samples")

plt.tight_layout()
plt.show()
# - 