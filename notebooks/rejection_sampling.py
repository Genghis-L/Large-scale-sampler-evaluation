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

# # Rejection Sampling Implementation

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
# -

# ## Rejection Sampling Implementation

# +
class RejectionSampling:
    """Implementation of rejection sampling.
    
    Rejection sampling is a basic Monte Carlo method that can sample from
    a target distribution by using a proposal distribution and accepting
    samples with probability proportional to the ratio of target to proposal.
    """
    
    def __init__(self, target_distribution, proposal_distribution, M=None):
        """Initialize rejection sampling.
        
        Args:
            target_distribution: Target distribution to sample from
            proposal_distribution: Proposal distribution to sample from
            M: Upper bound on target_density / proposal_density ratio. 
               If None, will be estimated from samples.
        """
        self.target = target_distribution
        self.proposal = proposal_distribution
        self.M = M
        
    def _estimate_M(self, num_samples=1000):
        """Estimate scaling factor M by sampling from proposal."""
        key = jax.random.PRNGKey(0)
        samples = self.proposal.sample(key, num_samples=num_samples)
        
        target_log_probs, _ = self.target.evaluate_log_density(samples, 0)
        proposal_log_probs, _ = self.proposal.evaluate_log_density(samples, 0)
        
        # Calculate the ratio and add some buffer
        ratios = jnp.exp(target_log_probs - proposal_log_probs)
        M = jnp.max(ratios) * 1.1  # Add 10% buffer
        
        print(f"Estimated M: {M:.4f}")
        return M
    
    def sample(self, key: Key, num_samples: int) -> Array:
        """Generate samples using rejection sampling.
        
        Args:
            key: PRNGKey for random number generation
            num_samples: Number of samples to generate
            
        Returns:
            Array of accepted samples
        """
        if self.M is None:
            self.M = self._estimate_M()
            
        # We'll need to generate more proposal samples than required
        # since some will be rejected
        expected_acceptance = 1.0 / self.M
        total_samples_needed = int(num_samples / expected_acceptance * 1.5)  # Add buffer
        
        accepted_samples = []
        num_accepted = 0
        attempts = 0
        max_attempts = total_samples_needed * 2  # Safeguard
        
        key, subkey = jax.random.split(key)
        proposal_samples = self.proposal.sample(subkey, num_samples=total_samples_needed)
        
        key, subkey = jax.random.split(key)
        uniform_samples = jax.random.uniform(subkey, shape=(total_samples_needed,))
        
        target_log_probs, _ = self.target.evaluate_log_density(proposal_samples, 0)
        proposal_log_probs, _ = self.proposal.evaluate_log_density(proposal_samples, 0)
        accept_probs = jnp.exp(target_log_probs - proposal_log_probs) / self.M
        
        # Use JAX to determine which samples to accept
        accepted_mask = uniform_samples <= accept_probs
        accepted_indices = jnp.where(accepted_mask)[0]
        
        # Take only the required number of samples
        actual_accepted = min(num_samples, accepted_indices.shape[0])
        accepted_samples = proposal_samples[accepted_indices[:actual_accepted]]
        
        # If we didn't get enough samples, try again with remaining
        remaining = num_samples - actual_accepted
        if remaining > 0:
            print(f"Only accepted {actual_accepted} samples, trying again for remaining {remaining}")
            key, subkey = jax.random.split(key)
            additional_samples = self.sample(subkey, remaining)
            accepted_samples = jnp.vstack([accepted_samples, additional_samples])
            
        acceptance_rate = actual_accepted / total_samples_needed
        print(f"Acceptance rate: {acceptance_rate:.4f}")
        
        return accepted_samples
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
    
    # Create the rejection sampler
    rejection_sampler = RejectionSampling(target_distribution, proposal_distribution)
    
    # Generate samples
    key = jax.random.PRNGKey(0)
    n_samples = 1000
    print(f"Generating {n_samples} samples using rejection sampling...")
    
    start_time = time.time()
    rejection_samples = rejection_sampler.sample(key, n_samples)
    elapsed_time = time.time() - start_time
    print(f"Sampling completed in {elapsed_time:.2f} seconds")
    
    # Generate ground truth samples for comparison
    key, subkey = jax.random.split(key)
    true_samples = target_distribution.sample(subkey, num_samples=n_samples)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # 1D marginal - x axis
    plt.subplot(1, 3, 1)
    sns.kdeplot(rejection_samples[:, 0], label="Rejection Sampling")
    sns.kdeplot(true_samples[:, 0], label="Ground Truth")
    plt.title("1D Marginal (x-axis)")
    plt.legend()
    
    # 1D marginal - y axis
    plt.subplot(1, 3, 2)
    sns.kdeplot(rejection_samples[:, 1], label="Rejection Sampling")
    sns.kdeplot(true_samples[:, 1], label="Ground Truth")
    plt.title("1D Marginal (y-axis)")
    plt.legend()
    
    # 2D scatter
    plt.subplot(1, 3, 3)
    plt.scatter(rejection_samples[:, 0], rejection_samples[:, 1], alpha=0.5, label="Rejection Sampling")
    plt.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, label="Ground Truth")
    plt.title("2D Samples")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
# - 