"""
1D Gaussian Example - Simple Demonstration

This script demonstrates the target distribution for CRAFT sampling.
It avoids the complex initialization issues with Haiku and flow models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add necessary paths
sys.path.append("/Users/kehanluo/Desktop/sampler workspace/annealed_flow_transport_Genghis")

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp

# Configuration
CONFIG = {
    'dim': 1,
    'num_steps': 10,
    'num_particles': 2000,
    'target_std': 2.0,
}
SEED = 0

# Define our normal distribution wrapper
class NormalDistributionWrapper:
    def __init__(self, dim, std, is_target=False):
        self.dim = dim
        self.std = std
        self.is_target = is_target
        self.log_Z = None  # Normalizing constant is 1 in log space
        self.can_sample = True
        self.mean_scale = 1.0  # For compatibility with other distributions
        
    def evaluate_log_density(self, x, step=0):
        """Compute log density of normal distribution."""
        log_prob = -0.5 * jnp.sum(x**2 / (self.std**2), axis=-1) - 0.5 * self.dim * jnp.log(2 * jnp.pi * self.std**2)
        return log_prob, jnp.zeros_like(log_prob)
        
    def log_prob(self, x):
        """Compute log probability."""
        log_prob = -0.5 * jnp.sum(x**2 / (self.std**2), axis=-1) - 0.5 * self.dim * jnp.log(2 * jnp.pi * self.std**2)
        return log_prob
        
    def sample(self, key, num_samples=1000):
        """Generate samples from the normal distribution."""
        return jax.random.normal(key, (num_samples, self.dim)) * self.std
        
    def visualise(self, samples, axes=None, show=False, prefix=''):
        """Simple visualization placeholder."""
        return {}

def main():
    """Main function to demonstrate the target distribution."""
    print("Setting up the target distribution...")
    
    # Target distribution - 1D Gaussian with specified variance
    target_dist = NormalDistributionWrapper(
        dim=CONFIG["dim"],
        std=CONFIG["target_std"],
        is_target=True
    )
    
    # Define initial distribution (standard normal)
    init_dist = NormalDistributionWrapper(
        dim=CONFIG["dim"],
        std=1.0,
        is_target=False
    )
    
    # Generate samples from both distributions
    key = jax.random.PRNGKey(SEED)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    target_samples = target_dist.sample(subkey1, num_samples=CONFIG["num_particles"])
    init_samples = init_dist.sample(subkey2, num_samples=CONFIG["num_particles"])
    
    # Plot results
    print("Plotting distributions...")
    plt.figure(figsize=(10, 6))
    
    # Plot density estimates
    sns.kdeplot(target_samples[:, 0], label=f"Target (std={CONFIG['target_std']})", color='blue')
    sns.kdeplot(init_samples[:, 0], label="Initial (std=1.0)", color='orange')
    
    plt.title("1D Gaussian Distributions")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # plt.savefig("1D-Gaussian-distributions.png")
    plt.show()
    
    print("Done!")
    print("\nTo implement CRAFT, you would need to:")
    print("1. Properly fix the import errors (see fix_moving_averages.py)")
    print("2. Properly implement flow models using Haiku's transform API")
    print("3. Adapt the code to use the correct paths for imports")
    print("\nSee the README.md for more details.")

if __name__ == "__main__":
    main() 