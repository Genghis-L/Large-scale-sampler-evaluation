# CRAFT Implementation Fix

This directory contains implementations of the CRAFT (Combined fLow-based transforms with Annealing) algorithm for various test scenarios.

## Import Error Fix

The original implementation had an import error:

```
ImportError: cannot import name 'moving_averages' from 'algorithms.common.eval_methods.utils'
```

### Solution

The issue was that the code was trying to import `moving_averages` from a module structure that didn't match the actual codebase. The fix involves:

1. Creating a `utils` directory within the `eval_methods` directory:
   ```
   mkdir -p algorithms/common/eval_methods/utils
   ```

2. Creating an `__init__.py` file in the `utils` directory with the necessary functions:
   ```python
   import jax.numpy as jnp
   
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
       # Implementation...
   
   def save_samples(cfg, logger, samples):
       # Implementation...
   
   def compute_reverse_ess(log_weights, eval_samples):
       # Implementation...
   ```

### Automated Fix

You can run the `fix_moving_averages.py` script to automatically apply the fix:

```bash
python experiments/CRAFT/fix_moving_averages.py --apply
```

## Flow Model Initialization Issue

After fixing the import error, you might encounter another error related to flow model initialization:

```
ValueError: All `hk.Module`s must be initialized inside an `hk.transform`.
```

This is because the flow models in the annealed_flow_transport package are implemented using Haiku, which requires special initialization. The solution involves:

1. Using `ComposedFlows` instead of directly instantiating flow models.
2. Properly initializing the flow model using Haiku's transformation API.

For example:

```python
import haiku as hk

# Define the flow model initialization function
def create_flow(config):
    flow = getattr(flows, config.type)(config)
    return flow

# Transform the function
transformed_flow = hk.transform(create_flow)

# Initialize parameters
key = jax.random.PRNGKey(SEED)
flow_samples = initial_sampler(key, (batch_size,))
flow_params = transformed_flow.init(key, craft_cfg.flow_config)
```

## Running the CRAFT Examples

To run the CRAFT examples, you need to:

1. Apply the fix for the import error as described above.
2. Ensure you have the necessary Python packages installed:
   ```bash
   pip install ott-jax
   ```

3. Set the PYTHONPATH to include the annealed_flow_transport_Genghis directory:
   ```bash
   PYTHONPATH=~/Desktop/sampler\ workspace/annealed_flow_transport_Genghis:$PYTHONPATH python experiments/CRAFT/1D-Gaussian.py
   ```

4. Modify the flow model initialization to use Haiku's transformation API.

## Alternative Implementation

For a simplified implementation that uses the annealed_flow_transport package directly, see `1D-Gaussian-simple.py`. This implementation requires further modifications to work with the specific flow models available in the package.

## Complete Solution

The complete solution to make CRAFT work in this codebase would involve:

1. Fixing the import error for `moving_averages`.
2. Properly initializing flow models using Haiku's transformation API.
3. Adapting the code to use the correct paths for imports from annealed_flow_transport.
4. Ensuring all dependencies are installed.

Given the complexity of these issues, we recommend:

1. Use the original annealed_flow_transport_Genghis implementation directly.
2. Create wrapper scripts that import from that codebase rather than trying to adapt the code to this codebase's structure.
3. Use the simplified implementation in `1D-Gaussian-simple.py` as a starting point for your experiments. 