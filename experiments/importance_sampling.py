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

# # Importance Sampling Implementation

# +
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from typing import Dict, Tuple, Callable, Any
from jaxtyping import PRNGKeyArray as Key, Array

from pdds.distributions import NormalDistributionWrapper, ChallengingTwoDimensionalMixture
from pdds.resampling import resampler
# -

# ## Importance Sampling Implementation

# +
class ImportanceSampling:
    """Implementation of importance sampling.
    
    Importance sampling is a technique for estimating properties of a target distribution,
    while sampling from a different, easier-to-sample-from distribution (the proposal).
    """
    
    def __init__(self, target_distribution, proposal_distribution):
        """Initialize importance sampling.
        
        Args:
            target_distribution: Target distribution to sample from
            proposal_distribution: Proposal distribution to sample from
        """
        self.target = target_distribution
        self.proposal = proposal_distribution
        
    def sample_with_weights(self, key: Key, num_samples: int) -> Dict:
        """Generate weighted samples using importance sampling.
        
        Args:
            key: PRNGKey for random number generation
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with:
                - samples: Array of samples
                - log_weights: Log weights for each sample
                - normalized_weights: Normalized weights (sum to 1)
                - ess: Effective sample size
                - ess_ratio: Effective sample size / num_samples
        """
        # Sample from proposal distribution
        proposal_samples = self.proposal.sample(key, num_samples=num_samples)
        
        # Compute importance weights: target pdf / proposal pdf
        target_log_probs, _ = self.target.evaluate_log_density(proposal_samples, 0)
        proposal_log_probs, _ = self.proposal.evaluate_log_density(proposal_samples, 0)
        log_weights = target_log_probs - proposal_log_probs
        
        # Normalize weights
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        normalized_weights = jnp.exp(log_weights_normalized)
        
        # Compute effective sample size
        ess = 1.0 / jnp.sum(normalized_weights**2)
        ess_ratio = ess / num_samples
        
        print(f"Effective sample size: {ess:.2f} ({ess_ratio:.4f} of total)")
        
        return {
            "samples": proposal_samples,
            "log_weights": log_weights,
            "normalized_weights": normalized_weights,
            "ess": ess,
            "ess_ratio": ess_ratio
        }
    
    def resample(self, key: Key, samples: Array, log_weights: Array, num_samples: int = None) -> Dict:
        """Resample from weighted samples to get unweighted samples.
        
        Args:
            key: PRNGKey for random number generation
            samples: Samples to resample from
            log_weights: Log weights for each sample
            num_samples: Number of samples to generate (default: same as input)
            
        Returns:
            Dictionary with resampled samples and weights
        """
        if num_samples is None:
            num_samples = samples.shape[0]
            
        # Use the resampler function from pdds
        return resampler(key, samples, log_weights)
    
    def sample(self, key: Key, num_samples: int) -> Array:
        """Generate unweighted samples by importance sampling followed by resampling.
        
        Args:
            key: PRNGKey for random number generation
            num_samples: Number of samples to generate
            
        Returns:
            Array of unweighted samples
        """
        key1, key2 = jax.random.split(key)
        result = self.sample_with_weights(key1, num_samples)
        resampled = self.resample(key2, result["samples"], result["log_weights"])
        return resampled["samples"]
    
    def estimate_expectation(self, f: Callable, key: Key, num_samples: int) -> Tuple[float, float]:
        """Estimate the expectation of a function under the target distribution.
        
        Args:
            f: Function to estimate expectation of
            key: PRNGKey for random number generation
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (estimate, standard_error)
        """
        result = self.sample_with_weights(key, num_samples)
        
        # Compute f(x) for each sample
        f_values = jax.vmap(f)(result["samples"])
        
        # Weighted average
        estimate = jnp.sum(result["normalized_weights"] * f_values)
        
        # Standard error
        squared_error = jnp.sum(result["normalized_weights"] * (f_values - estimate)**2)
        standard_error = jnp.sqrt(squared_error / num_samples)
        
        return estimate, standard_error
# -

# ## Example Usage

# +
# Example usage
if __name__ == "__main__":
    # Set up a simple target and proposal
    dim = 2
    mean_scale = 3.0
    
    # Create distributions
    target_distribution = ChallengingTwoDimensionalMixture(mean_scale=mean_scale, dim=dim, is_target=True)
    proposal_distribution = NormalDistributionWrapper(mean=0.0, scale=5.0, dim=dim)
    
    # Create the importance sampler
    importance_sampler = ImportanceSampling(target_distribution, proposal_distribution)
    
    # Generate samples
    key = jax.random.PRNGKey(0)
    n_samples = 2000
    print(f"Generating {n_samples} samples using importance sampling...")
    
    start_time = time.time()
    importance_result = importance_sampler.sample_with_weights(key, n_samples)
    key, subkey = jax.random.split(key)
    resampled_result = importance_sampler.resample(subkey, importance_result["samples"], importance_result["log_weights"])
    importance_samples = resampled_result["samples"]
    elapsed_time = time.time() - start_time
    print(f"Sampling completed in {elapsed_time:.2f} seconds")
    
    # Generate ground truth samples for comparison
    key, subkey = jax.random.split(key)
    true_samples = target_distribution.sample(subkey, num_samples=n_samples)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # 1D marginal - x axis
    plt.subplot(2, 2, 1)
    sns.kdeplot(importance_samples[:, 0], label="Importance Sampling (Resampled)")
    sns.kdeplot(true_samples[:, 0], label="Ground Truth")
    plt.title("1D Marginal (x-axis)")
    plt.legend()
    
    # 1D marginal - y axis
    plt.subplot(2, 2, 2)
    sns.kdeplot(importance_samples[:, 1], label="Importance Sampling (Resampled)")
    sns.kdeplot(true_samples[:, 1], label="Ground Truth")
    plt.title("1D Marginal (y-axis)")
    plt.legend()
    
    # 2D scatter - raw weighted samples with alpha proportional to weight
    plt.subplot(2, 2, 3)
    max_weight = jnp.max(importance_result["normalized_weights"])
    alphas = importance_result["normalized_weights"] / max_weight
    alphas = jnp.clip(alphas, 0.05, 0.9)  # Clip for better visualization
    
    for i in range(min(n_samples, 500)):  # Limit to 500 points for clearer visualization
        plt.scatter(
            importance_result["samples"][i, 0], 
            importance_result["samples"][i, 1], 
            alpha=float(alphas[i]), 
            color='blue',
            s=20
        )
    plt.scatter([], [], color='blue', label="Weighted Samples")  # For legend
    plt.title("Weighted Samples (alpha ∝ weight)")
    plt.legend()
    
    # 2D scatter - resampled
    plt.subplot(2, 2, 4)
    plt.scatter(importance_samples[:, 0], importance_samples[:, 1], alpha=0.5, label="Resampled", color='green')
    plt.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, label="Ground Truth", color='red')
    plt.title("Resampled vs Ground Truth")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Bonus: estimate an expectation (e.g., mean of first component)
    def f(x):
        return x[0]
    
    est, se = importance_sampler.estimate_expectation(f, key, n_samples)
    print(f"Estimated mean of first component: {est:.4f} ± {se:.4f}")
    true_mean = jnp.mean(true_samples[:, 0])
    print(f"True mean of first component: {true_mean:.4f}")
# - 