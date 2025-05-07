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

# # 2D Hard Gaussian Mixture - Rejection Sampling

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

# ## 2. Rejection Sampling Implementation

# +
def rejection_sampling(
    proposal_dist,
    target_log_prob_fn,
    scaling_factor,
    num_samples,
    key,
    max_iterations=10,
    batch_size_multiplier=10
):
    """
    Performs rejection sampling to generate samples from a target distribution.
    
    Args:
        proposal_dist: Distribution to sample from
        target_log_prob_fn: Function to compute target log probability
        scaling_factor: Scaling factor for rejection test (M in MH algorithm)
        num_samples: Number of samples to generate
        key: JAX random key
        max_iterations: Maximum number of iterations to try
        batch_size_multiplier: Multiplier for batch size to process at once
        
    Returns:
        Accepted samples from the target distribution
    """
    accepted_samples = []
    total_proposed = 0
    total_accepted = 0
    
    # Process in batches for efficiency
    batch_size = int(num_samples * batch_size_multiplier)
    
    for i in range(max_iterations):
        # Split the random key
        key, proposal_key, uniform_key = jax.random.split(key, 3)
        
        # Generate proposals from the proposal distribution
        proposed_samples = proposal_dist.sample(proposal_key, batch_size)
        total_proposed += batch_size
        
        # Evaluate log densities
        log_q_density, _ = proposal_dist.evaluate_log_density(proposed_samples, 0)
        log_p_density, _ = target_log_prob_fn(proposed_samples, 0)
        
        # Rejection test
        log_acceptance_ratio = log_p_density - log_q_density - jnp.log(scaling_factor)
        log_u = jnp.log(jax.random.uniform(uniform_key, shape=(batch_size,)))
        
        # Determine accepted samples
        accepted_mask = log_u <= log_acceptance_ratio
        accepted_from_batch = proposed_samples[accepted_mask]
        total_accepted += accepted_from_batch.shape[0]
        
        # Store accepted samples
        accepted_samples.append(accepted_from_batch)
        
        # Check if we have enough samples
        total_accepted_samples = sum(x.shape[0] for x in accepted_samples)
        if total_accepted_samples >= num_samples:
            # Concatenate all accepted samples
            all_accepted = jnp.concatenate(accepted_samples, axis=0)
            # Return only the requested number of samples
            return all_accepted[:num_samples], total_proposed, total_accepted
    
    # If we've reached the maximum iterations without getting enough samples,
    # return what we have and print a warning
    print(f"Warning: Only collected {total_accepted} samples after {max_iterations} iterations")
    all_accepted = jnp.concatenate(accepted_samples, axis=0) if accepted_samples else jnp.zeros((0, proposal_dist.dim))
    
    # If we have at least some samples, duplicate to reach the requested number
    if all_accepted.shape[0] > 0:
        # Repeat the samples to reach the desired count
        indices = jnp.mod(jnp.arange(num_samples), all_accepted.shape[0])
        return all_accepted[indices], total_proposed, total_accepted
    else:
        # No samples were accepted, return zeros as a fallback
        return jnp.zeros((num_samples, proposal_dist.dim)), total_proposed, total_accepted

def estimate_scaling_factor(proposal_dist, target_dist, num_samples=1000, safety_factor=1.5):
    """
    Estimate the scaling factor for rejection sampling.
    
    Args:
        proposal_dist: Proposal distribution
        target_dist: Target distribution
        num_samples: Number of samples to use for estimation
        safety_factor: Additional safety factor to multiply the estimate by
        
    Returns:
        Estimated scaling factor M
    """
    # Generate samples from the proposal distribution
    key = jax.random.PRNGKey(0)
    samples = proposal_dist.sample(key, num_samples)
    
    # Compute density ratios
    log_p_density, _ = target_dist.evaluate_log_density(samples, 0)
    log_q_density, _ = proposal_dist.evaluate_log_density(samples, 0)
    log_ratio = log_p_density - log_q_density
    
    # Find the maximum ratio and apply safety factor
    max_ratio = jnp.exp(jnp.max(log_ratio)) * safety_factor
    
    return max_ratio

def plot_gaussian_mixture_results(target_samples, samples, mean_scale, label="Rejection Sampling", metrics=None):
    """
    Creates a visualization for Gaussian mixture results with 
    2D scatter plot and 1D marginals.
    
    Args:
        target_samples: Samples from target distribution
        samples: Samples from rejection sampling
        mean_scale: Scale factor of Gaussian mixture means
        label: Label for the method
        metrics: Dictionary with metrics to display
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 2D scatter
    axs[0].scatter(
        target_samples[:, 0], 
        target_samples[:, 1], 
        alpha=0.3, 
        label="Target", 
        s=10,
        color="red"
    )
    axs[0].scatter(
        samples[:, 0], 
        samples[:, 1], 
        alpha=0.6, 
        label=label, 
        s=10
    )
    
    axs[0].set_title(f"Hard 2D Gaussian Mixture (mean_scale={mean_scale})")
    if metrics:
        subtitle = f"Acceptance Rate: {metrics['acceptance_rate']:.2f}%, Samples Needed: {metrics['total_proposed']}"
        axs[0].text(0.5, -0.1, subtitle, transform=axs[0].transAxes, ha='center')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')
    
    # Plot 1D marginal x-axis
    sns.kdeplot(target_samples[:, 0], label="Target", ax=axs[1], color="red")
    sns.kdeplot(samples[:, 0], label=label, ax=axs[1])
    axs[1].set_title("1D Marginal (x-axis)")
    axs[1].set_xlabel("x")
    axs[1].legend()
    
    # Plot 1D marginal y-axis
    sns.kdeplot(target_samples[:, 1], label="Target", ax=axs[2], color="red")
    sns.kdeplot(samples[:, 1], label=label, ax=axs[2])
    axs[2].set_title("1D Marginal (y-axis)")
    axs[2].set_xlabel("y")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
# -

# ## 3. Testing with Different Mean Scales

# +
def run_rejection_sampling_experiment(mean_scale, num_samples=2000):
    """
    Run a rejection sampling experiment for the specified mean scale.
    
    Args:
        mean_scale: Scale factor for Gaussian mixture means
        num_samples: Number of samples to generate
    """
    global key
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
    
    # Estimate scaling factor
    scaling_factor = estimate_scaling_factor(proposal_dist, target_dist)
    print(f"Estimated scaling factor: {scaling_factor:.4f}")
    
    # Run rejection sampling
    key, subkey1, subkey2 = jax.random.split(key, 3)
    start_time = time.time()
    samples, total_proposed, total_accepted = rejection_sampling(
        proposal_dist,
        target_dist.evaluate_log_density,
        scaling_factor,
        num_samples,
        subkey1
    )
    end_time = time.time()
    
    # Generate true target samples for comparison
    target_samples = target_dist.sample(subkey2, num_samples=num_samples)
    
    # Calculate metrics
    acceptance_rate = (total_accepted / total_proposed) * 100 if total_proposed > 0 else 0
    metrics = {
        'acceptance_rate': acceptance_rate,
        'total_proposed': total_proposed,
        'time_taken': end_time - start_time
    }
    
    # Print statistics
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    print(f"Total proposed samples: {total_proposed}")
    print(f"Total accepted samples: {total_accepted}")
    print(f"Time taken: {metrics['time_taken']:.2f} seconds")
    
    # Plot results
    plot_gaussian_mixture_results(
        target_samples,
        samples,
        mean_scale,
        metrics=metrics
    )
    
    return samples, metrics, target_samples

# Run experiments with different mean scales
mean_scales = [1.0, 2.0, 3.0, 4.0, 5.0]
results = {}

for scale in mean_scales:
    samples, metrics, target_samples = run_rejection_sampling_experiment(scale)
    results[scale] = {
        'samples': samples,
        'metrics': metrics,
        'target_samples': target_samples
    }
# -

# ## 4. Comparing Acceptance Rates Across Mean Scales

# +
# Plot acceptance rate comparison
plt.figure(figsize=(10, 6))
acceptance_rates = [results[scale]['metrics']['acceptance_rate'] for scale in mean_scales]
plt.bar(mean_scales, acceptance_rates)
plt.xlabel('Mean Scale')
plt.ylabel('Acceptance Rate (%)')
plt.title('Rejection Sampling Performance vs. Mean Scale')
plt.xticks(mean_scales)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for i, v in enumerate(acceptance_rates):
    plt.text(mean_scales[i], v + 0.5, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()

# Plot computation time
plt.figure(figsize=(10, 6))
times = [results[scale]['metrics']['time_taken'] for scale in mean_scales]
plt.bar(mean_scales, times)
plt.xlabel('Mean Scale')
plt.ylabel('Computation Time (seconds)')
plt.title('Rejection Sampling Computation Time vs. Mean Scale')
plt.xticks(mean_scales)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for i, v in enumerate(times):
    plt.text(mean_scales[i], v + 0.1, f"{v:.2f}s", ha='center')

plt.tight_layout()
plt.show()

# Plot number of proposals needed
plt.figure(figsize=(10, 6))
proposals = [results[scale]['metrics']['total_proposed'] for scale in mean_scales]
proposals = [p/1000 for p in proposals]  # Convert to thousands for better readability
plt.bar(mean_scales, proposals)
plt.xlabel('Mean Scale')
plt.ylabel('Proposals Needed (thousands)')
plt.title('Number of Proposals Required vs. Mean Scale')
plt.xticks(mean_scales)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for i, v in enumerate(proposals):
    plt.text(mean_scales[i], v + 0.5, f"{v:.1f}k", ha='center')

plt.tight_layout()
plt.show()
# -

# ## 5. Summary and Conclusion

# +
# Print a summary of the experiments
print("=" * 80)
print("Rejection Sampling for Hard 2D Gaussian Mixture - Summary")
print("=" * 80)
print("Sampling Performance Across Different Mean Scales:")
for scale in mean_scales:
    metrics = results[scale]['metrics']
    print(f"- Mean Scale {scale:.1f}: Acceptance Rate = {metrics['acceptance_rate']:.2f}%, "
          f"Time = {metrics['time_taken']:.2f}s, "
          f"Proposals = {metrics['total_proposed']}")
print("-" * 80)
print("Observations:")
print("1. As the mean scale increases, the acceptance rate decreases dramatically")
print("2. The number of proposal samples needed grows exponentially with mean scale")
print("3. Computation time increases with mean scale due to lower acceptance rates")
print("4. For high mean scales, rejection sampling becomes extremely inefficient")
print("=" * 80)

"""
Conclusion:
Rejection sampling becomes extremely inefficient for sampling from
multimodal distributions with well-separated modes. The required number of
proposals increases dramatically as the modes move further apart, making
the method impractical for complex target distributions. This demonstrates
the fundamental limitation of simple Monte Carlo methods for multimodal
distributions and highlights the need for more sophisticated approaches like
MCMC, SMC, or normalizing flows for such challenging sampling problems.
"""
# -