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

# +
# %load_ext autoreload
# %autoreload 2

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

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

from functools import partial

from jaxtyping import PRNGKeyArray as Key, Array
import typing as tp
from check_shapes import check_shapes

from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture
from pdds.sde import SDE, guidance_loss, dsm_loss
from pdds.smc_problem import SMCProblem
from pdds.potentials import NNApproximatedPotential
from pdds.utils.shaping import broadcast
from pdds.utils.jax import (
    _get_key_iter,
    x_gradient_stateful_parametrised,
)

# -

# ## Sample Standard Gaussian with NUTS

# +
# Set the random seed for reproducibility
numpyro.set_host_device_count(1)
rng_key = jax.random.PRNGKey(0)

# Define your potential function (negative log probability)
# This example uses a 2D multivariate normal with correlation
def potential_fn(z):
    # Example: Multivariate Normal potential
    # You can replace this with your own potential function
    mu = jnp.array([0., 0.])
    sigma = jnp.array([[1.0, 0.0],
                        [0.0, 1.0]])
    precision = jnp.linalg.inv(sigma)
    delta = z - mu
    return 0.5 * jnp.dot(delta, jnp.dot(precision, delta))

# Define model using the potential function
def model():
    # Use sample statements with proper distributions instead of ImproperUniform
    # Start with normal distributions for each dimension
    z = numpyro.sample('z', dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=jnp.eye(2)
    ))
    # Calculate the proposal log density
    proposal_log_density = dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=jnp.eye(2)
    ).log_prob(z)
    potential_value = potential_fn(z)

    # Add the correction factor (target - proposal)
    numpyro.factor('potential', -potential_value - proposal_log_density)
    return z

# Set up the NUTS sampler
nuts_kernel = NUTS(model)

# Run MCMC
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(rng_key)

# Get the samples
samples = mcmc.get_samples()
z1_samples = samples['z'][:, 0]
z2_samples = samples['z'][:, 1]
z_samples = jnp.column_stack([z1_samples, z2_samples])

# Print some summary statistics
print("Sample mean:", jnp.mean(z_samples, axis=0))
print("Sample covariance:\n", jnp.cov(z_samples, rowvar=False))

# Visualize the samples
plt.figure(figsize=(6, 4))
plt.scatter(z_samples[:, 0], z_samples[:, 1], alpha=0.5)
plt.title('NUTS Samples from the Potential Function')
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.grid(True)
plt.axis('equal')

# Trace plots
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(z_samples[:, 0])
plt.title('Trace of z[0]')
plt.xlabel('Iteration')
plt.subplot(1, 2, 2)
plt.plot(z_samples[:, 1])
plt.title('Trace of z[1]')
plt.xlabel('Iteration')
plt.tight_layout()
plt.show()

# -

# ## Gaussian Mixture Example

# +
# global variables
dim=2
mean_scale = 0.8
sigma=1.0
t_0=0.0
t_f=1.0
num_steps=24
num_particles=2000
beta = 0.2

# INSTANTIATE OBJECTS

# Instantiate key iterator
key = jax.random.PRNGKey(seed=0)
key_iter = _get_key_iter(key)

# Instantiate target
# target_distribution = NormalDistributionWrapper(mean=2.75, scale=0.25, dim=dim, is_target=True)
# target_distribution_intermediate = NormalDistributionWrapper(mean=2.75, scale=0.25/beta, dim=dim, is_target=True)
target_distribution = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True)
target_distribution_intermediate = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True, beta=beta)

# +
# Set the random seed for reproducibility
numpyro.set_host_device_count(1)
rng_key = jax.random.PRNGKey(0)

def mixture_model():
    # Start with a sample statement to define the variable
    z = numpyro.sample('z', dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=jnp.eye(2) * 6
    ))

    # Calculate your log density
    z_batched = z.reshape(1, -1)
    log_density, _ = target_distribution.evaluate_log_density(z_batched, 0)
    log_density = log_density[0]  # Remove batch dimension

    # Subtract the proposal log density to cancel it out
    proposal_log_density = dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=jnp.eye(2) * 6
    ).log_prob(z)

    # Add the correction factor
    numpyro.factor('density_factor', log_density - proposal_log_density)
    return z

# Set up the NUTS sampler
nuts_kernel = NUTS(mixture_model)

# Run MCMC
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=4000)
mcmc.run(rng_key)

# Get the samples
samples = mcmc.get_samples()
z1_samples = samples['z'][:, 0]
z2_samples = samples['z'][:, 1]
z_samples = jnp.column_stack([z1_samples, z2_samples])

# Print some summary statistics
print("Sample mean:", jnp.mean(z_samples, axis=0))
print("Sample covariance:\n", jnp.cov(z_samples, rowvar=False))

n_plot_samples = int(num_particles)
idx = jnp.arange(int(num_particles))
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)

# Visualize the samples
plt.figure(figsize=(10, 8))
fig = plt.figure()
ax = fig.gca()
sns.kdeplot(z_samples[:num_particles, 0], ax=ax, label="NUTS")
sns.kdeplot(target_samples[:, 0], ax=ax, label="Target")
plt.legend()
plt.show()
plt.close(fig)

# Trace plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(z_samples[:, 0])
plt.title('Trace of z[0]')
plt.xlabel('Iteration')
plt.subplot(1, 2, 2)
plt.plot(z_samples[:, 1])
plt.title('Trace of z[1]')
plt.xlabel('Iteration')
plt.tight_layout()
plt.show()


# +
# Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/ReplicaExchangeMC
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions

# Create a TensorFlow version of your mixture distribution
def create_tensorflow_mixture():
    # Define the components based on your JAX implementation
    component_locs = [
        [3.0, 0.0],   # Mode 1
        [0.0, 3.0],   # Mode 2
        [-2.5, 0.0],  # Mode 3
        [0.0, -2.5],  # Mode 4
        [2.0, 3.0],   # Mode 5
        [3.0, 2.0]    # Mode 6
    ]

    # Define covariance matrices
    component_covs = [
        [[0.7, 0.0], [0.0, 0.05]],  # Mode 1
        [[0.05, 0.0], [0.0, 0.7]],  # Mode 2
        [[0.7, 0.0], [0.0, 0.05]],  # Mode 3
        [[0.05, 0.0], [0.0, 0.7]],  # Mode 4
        [[1.0, 0.95], [0.95, 1.0]],  # Mode 5
        [[1.0, 0.95], [0.95, 1.0]]   # Mode 6
    ]

    # Convert to tensors
    locs = tf.convert_to_tensor(component_locs, dtype=tf.float32)
    covs = tf.convert_to_tensor(component_covs, dtype=tf.float32)

    # Component weights (equal weighting)
    probs = [1/6.] * 6

    # Create mixture distribution
    mixture = tfd.Mixture(
        cat=tfd.Categorical(probs=probs),
        components=[
            tfd.MultivariateNormalFullCovariance(loc=locs[i], covariance_matrix=covs[i])
            for i in range(6)
        ]
    )

    return mixture

# Create the TensorFlow mixture
tf_mixture = create_tensorflow_mixture()

# Setup for ReplicaExchangeMC
dtype = np.float32
inverse_temperatures = 0.2**tf.range(4, dtype=dtype)

# Step size
step_size = 0.075 / tf.reshape(tf.sqrt(inverse_temperatures), shape=(4, 1))

# Define kernel maker function
def make_kernel_fn(target_log_prob_fn):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=1
    )

# Create ReplicaExchangeMC kernel
remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=tf_mixture.log_prob,
    inverse_temperatures=inverse_temperatures,
    make_kernel_fn=make_kernel_fn
)

# Initial state
initial_state = tf.ones(2, dtype=dtype) * tf.constant([3.0, 0.0], dtype=dtype)

# Run sampling
print("Starting ReplicaExchangeMC sampling with TensorFlow mixture...")
samples = tfp.mcmc.sample_chain(
    num_results=200,
    current_state=initial_state,
    kernel=remc,
    trace_fn=None,
    num_burnin_steps=50
)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
plt.title('ReplicaExchangeMC Samples from TensorFlow Mixture')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()

# -


