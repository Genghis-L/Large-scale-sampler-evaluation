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

# # 2D Hard Gaussian Mixture - Importance Sampling

# ## 1. Setup and Imports

# +
# %load_ext autoreload
# %autoreload 2

import os
import time
import tqdm
import typing as tp
from functools import partial

# Set available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# JAX imports
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray as Key, Array
from check_shapes import check_shapes

# Data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PDDS modules
from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture

# Define seed for reproducibility
SEED = 0
key = jax.random.PRNGKey(seed=SEED)
# -

# ## 2. Importance Sampling Implementation

# +
@check_shapes("return[0]: [b, d]", "return[1]: [b]", "return[2]: [b]", "return[3]: []")
def importance_sampling(
    proposal_dist,
    target_dist,
    num_samples: int,
    key: Key,
):
    """
    Performs importance sampling using the given proposal and target distributions.
    
    Args:
        proposal_dist: Distribution to sample from
        target_dist: Target distribution we want to estimate
        num_samples: Number of samples to generate
        key: JAX random key
        
    Returns:
        Tuple of (samples, log_weights, normalized_weights, ess)
    """
    # Generate samples from the proposal distribution
    samples = proposal_dist.sample(key, num_samples)
    
    # Compute log densities under both distributions
    log_q_density, _ = proposal_dist.evaluate_log_density(samples, 0)
    log_p_density, _ = target_dist.evaluate_log_density(samples, 0)
    
    # Calculate importance weights (in log-space for numerical stability)
    log_weights = log_p_density - log_q_density
    
    # Normalize the weights
    max_log_weight = jnp.max(log_weights)
    normalized_log_weights = log_weights - max_log_weight
    normalized_weights = jnp.exp(normalized_log_weights)
    normalized_weights = normalized_weights / jnp.sum(normalized_weights)
    
    # Calculate effective sample size (ESS)
    ess = 1.0 / jnp.sum(normalized_weights**2) / num_samples
    
    return samples, log_weights, normalized_weights, ess

def resample(samples, normalized_weights, key, num_samples=None):
    """
    Resample from weighted samples to get unweighted samples.
    
    Args:
        samples: Original weighted samples
        normalized_weights: Normalized importance weights
        key: JAX random key
        num_samples: Number of samples to generate (defaults to original sample count)
        
    Returns:
        Resampled samples with uniform weights
    """
    if num_samples is None:
        num_samples = samples.shape[0]
    
    # Perform multinomial resampling
    indices = jax.random.choice(
        key,
        samples.shape[0],
        shape=(num_samples,),
        replace=True,
        p=normalized_weights
    )
    
    resampled = samples[indices]
    return resampled

def plot_gaussian_mixture_results(target_samples, samples, normalized_weights, mean_scale, label="Importance Sampling"):
    """
    Creates a visualization for Gaussian mixture results with 
    2D scatter plot and 1D marginals.
    
    Args:
        target_samples: Samples from target distribution
        samples: Samples from approximation method
        normalized_weights: Sample weights for importance sampling
        mean_scale: Scale factor of Gaussian mixture means
        label: Label for the method
    """
    # Create weighted and resampled versions
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 2D scatter weighted samples
    scatter_size = 10 + 90 * normalized_weights / jnp.max(normalized_weights)
    scatter = axs[0, 0].scatter(
        samples[:, 0], 
        samples[:, 1], 
        alpha=0.6, 
        label=f"Weighted {label}", 
        s=scatter_size,
        c=normalized_weights,
        cmap="viridis"
    )
    axs[0, 0].scatter(
        target_samples[:, 0], 
        target_samples[:, 1], 
        alpha=0.3, 
        label="Target", 
        s=10,
        color="red"
    )
    
    axs[0, 0].set_title(f"Weighted Samples - Hard 2D Gaussian Mixture (mean_scale={mean_scale})")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')
    
    # Add colorbar for weight visualization
    plt.colorbar(scatter, ax=axs[0, 0], label="Weight")
    
    # Plot 1D marginal x-axis (weighted)
    sns.kdeplot(x=target_samples[:, 0], label="Target", ax=axs[0, 1], color="red")
    sns.kdeplot(x=samples[:, 0], label=f"Weighted {label}", ax=axs[0, 1], 
                weights=normalized_weights)
    axs[0, 1].set_title("1D Marginal (x-axis) - Weighted")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].legend()
    
    # Plot 1D marginal y-axis (weighted)
    sns.kdeplot(x=target_samples[:, 1], label="Target", ax=axs[0, 2], color="red")
    sns.kdeplot(x=samples[:, 1], label=f"Weighted {label}", ax=axs[0, 2], 
                weights=normalized_weights)
    axs[0, 2].set_title("1D Marginal (y-axis) - Weighted")
    axs[0, 2].set_xlabel("y")
    axs[0, 2].legend()
    
    # Generate resampled points
    key = jax.random.PRNGKey(42)
    resampled_points = resample(samples, normalized_weights, key)
    
    # Plot 2D scatter resampled
    axs[1, 0].scatter(
        resampled_points[:, 0], 
        resampled_points[:, 1], 
        alpha=0.6, 
        label=f"Resampled {label}", 
        s=10
    )
    axs[1, 0].scatter(
        target_samples[:, 0], 
        target_samples[:, 1], 
        alpha=0.3, 
        label="Target", 
        s=10,
        color="red"
    )
    
    axs[1, 0].set_title(f"Resampled Points - Hard 2D Gaussian Mixture (mean_scale={mean_scale})")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].axis('equal')
    
    # Plot 1D marginal x-axis (resampled)
    sns.kdeplot(target_samples[:, 0], label="Target", ax=axs[1, 1], color="red")
    sns.kdeplot(resampled_points[:, 0], label=f"Resampled {label}", ax=axs[1, 1])
    axs[1, 1].set_title("1D Marginal (x-axis) - Resampled")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].legend()
    
    # Plot 1D marginal y-axis (resampled)
    sns.kdeplot(target_samples[:, 1], label="Target", ax=axs[1, 2], color="red")
    sns.kdeplot(resampled_points[:, 1], label=f"Resampled {label}", ax=axs[1, 2])
    axs[1, 2].set_title("1D Marginal (y-axis) - Resampled")
    axs[1, 2].set_xlabel("y")
    axs[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
# -

# ## 3. Testing with Different Mean Scales

# +
def run_importance_sampling_experiment(mean_scale, num_samples=2000, key=None):
    """
    Run an importance sampling experiment for the specified mean scale.
    
    Args:
        mean_scale: Scale factor for Gaussian mixture means
        num_samples: Number of samples to generate
        key: JAX random key
    """
    if key is None:
        key = jax.random.PRNGKey(0)
        
    print(f"\n--- Running experiment with mean_scale={mean_scale} ---")
    
    # Create target distribution
    target_dist = ChallengingTwoDimensionalMixture(
        mean_scale=mean_scale, 
        dim=2, 
        is_target=True
    )
    
    # Create proposal distribution (wider standard normal)
    proposal_dist = NormalDistributionWrapper(
        mean=0.0, 
        scale=max(4.0, mean_scale*1.5),  # Scale proposal based on target scale
        dim=2, 
        is_target=False
    )
    
    # Run importance sampling
    key, subkey1, subkey2 = jax.random.split(key, 3)
    samples, log_weights, normalized_weights, ess = importance_sampling(
        proposal_dist, 
        target_dist, 
        num_samples, 
        subkey1
    )
    
    # Generate true target samples for comparison
    target_samples = target_dist.sample(subkey2, num_samples=num_samples)
    
    # Print statistics
    print(f"Effective Sample Size (ESS): {ess*100:.2f}%")
    print(f"Min weight: {jnp.min(normalized_weights):.8f}")
    print(f"Max weight: {jnp.max(normalized_weights):.8f}")
    
    # Plot results
    plot_gaussian_mixture_results(
        target_samples, 
        samples, 
        normalized_weights,
        mean_scale
    )
    
    return samples, log_weights, normalized_weights, ess, target_samples

# Run experiments with different mean scales
mean_scales = [1.0, 2.0, 3.0, 4.0, 5.0]
results = {}

# Create a fresh key for the experiments
experiment_key = jax.random.PRNGKey(SEED)

for scale in mean_scales:
    experiment_key, subkey = jax.random.split(experiment_key)
    samples, log_weights, normalized_weights, ess, target_samples = run_importance_sampling_experiment(scale, key=subkey)
    results[scale] = {
        'samples': samples,
        'log_weights': log_weights,
        'normalized_weights': normalized_weights,
        'ess': ess,
        'target_samples': target_samples
    }
# -

# ## 4. Comparing ESS Across Mean Scales

# +
# Plot ESS comparison across mean scales
plt.figure(figsize=(10, 6))
ess_values = [results[scale]['ess'] * 100 for scale in mean_scales]
plt.bar(mean_scales, ess_values)
plt.xlabel('Mean Scale')
plt.ylabel('Effective Sample Size (%)')
plt.title('Importance Sampling Performance vs. Mean Scale')
plt.xticks(mean_scales)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for i, v in enumerate(ess_values):
    plt.text(mean_scales[i], v + 0.5, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()
# -

# ## 5. Summary and Conclusion

# +
# Print a summary of the experiments
print("=" * 80)
print("Importance Sampling for Hard 2D Gaussian Mixture - Summary")
print("=" * 80)
print("Sampling Performance Across Different Mean Scales:")
for scale in mean_scales:
    print(f"- Mean Scale {scale:.1f}: ESS = {results[scale]['ess']*100:.2f}%")
print("-" * 80)
print("Observations:")
print("1. As the mean scale increases, the mixture components move farther apart")
print("2. Importance sampling becomes less efficient with higher mean scales")
print("3. This demonstrates the challenge of multimodal distributions for sampling")
print("=" * 80)

"""
Conclusion:
For 2D Gaussian mixtures with well-separated modes, importance sampling
becomes inefficient due to the proposal distribution's inability to
adequately cover all modes. The effective sample size decreases dramatically
as mode separation increases, highlighting the need for more sophisticated
sampling techniques for multimodal distributions.
"""
# -