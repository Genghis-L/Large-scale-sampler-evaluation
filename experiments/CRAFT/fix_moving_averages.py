"""
Fix for the moving_averages import error in CRAFT implementation.

This script demonstrates the solution to the import error:
ImportError: cannot import name 'moving_averages' from 'algorithms.common.eval_methods.utils'

The fix involves creating the utils directory within eval_methods and implementing
the moving_averages function.
"""

import os
import sys
import shutil
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fix moving_averages import error')
    parser.add_argument('--apply', action='store_true', help='Apply the fix')
    args = parser.parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define paths
    eval_methods_dir = os.path.join(base_dir, 'algorithms', 'common', 'eval_methods')
    utils_dir = os.path.join(eval_methods_dir, 'utils')
    utils_init_file = os.path.join(utils_dir, '__init__.py')
    utils_file = os.path.join(eval_methods_dir, 'utils.py')
    
    # Check if the files exist
    if not os.path.exists(eval_methods_dir):
        print(f"Error: {eval_methods_dir} does not exist.")
        return 1
    
    # Print current state
    print(f"Base directory: {base_dir}")
    print(f"Eval methods directory: {eval_methods_dir}")
    print(f"Utils directory to create: {utils_dir}")
    print(f"Utils file: {utils_file}")
    
    if os.path.exists(utils_file):
        print(f"Utils file exists: {utils_file}")
    else:
        print(f"Utils file does not exist: {utils_file}")
    
    if os.path.exists(utils_dir):
        print(f"Utils directory already exists: {utils_dir}")
    else:
        print(f"Utils directory does not exist: {utils_dir}")
    
    # If not applying the fix, just print what would be done
    if not args.apply:
        print("\nTo apply the fix, run with --apply flag")
        return 0
    
    # Apply the fix
    print("\nApplying the fix...")
    
    # Create the utils directory if it doesn't exist
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"Created directory: {utils_dir}")
    
    # Create the __init__.py file with the necessary functions
    with open(utils_init_file, 'w') as f:
        f.write("""import jax.numpy as jnp

from utils.path_utils import project_path


def moving_averages(dictionary, window_size=5):
    mov_avgs = {}
    for key, value in dictionary.items():
        try:
            if not 'mov_avg' in key:
                mov_avgs[f'{key}_mov_avg'] = [jnp.mean(jnp.array(value[-min(len(value), window_size):]), axis=0)]
        except:
            pass
    return mov_avgs


def extract_last_entry(dictionary):
    last_entries = {}
    for key, value in dictionary.items():
        try:
            last_entries[key] = value[-min(len(value), 1)]
        except:
            pass
    return last_entries


def save_samples(cfg, logger, samples):
    if len(logger['KL/elbo']) > 1:
        if logger['KL/elbo'][-1] >= jnp.max(jnp.array(logger['KL/elbo'][:-1])):
            jnp.save(project_path(f'{cfg.log_dir}/samples_{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}'), samples)
        else:
            return
    else:
        jnp.save(project_path(f'{cfg.log_dir}/samples_{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}'),
                 samples)


def compute_reverse_ess(log_weights, eval_samples):
    # Subtract the maximum log weight for numerical stability
    max_log_weight = jnp.max(log_weights)
    stable_log_weights = log_weights - max_log_weight

    # Compute the importance weights in a numerically stable way
    is_weights = jnp.exp(stable_log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = jnp.sum(is_weights)
    sum_is_weights_squared = jnp.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (eval_samples * sum_is_weights_squared)

    return ess
""")
        print(f"Created file: {utils_init_file}")
    
    print("\nFix applied successfully!")
    print("\nYou should now be able to run the CRAFT implementation without the moving_averages import error.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 