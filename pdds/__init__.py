"""
Particle-based deterministic sampling (PDDS) package.
"""

# Fix for check_shapes and tensorflow-probability compatibility issues
import sys
from importlib.util import find_spec

# Only apply the patch if check_shapes is installed
if find_spec("check_shapes"):
    # Monkeypatch check_shapes.integration.tfp to avoid the __internal__ error
    import types
    import check_shapes
    
    # Create a mock install_tfp_integration function that does nothing
    def mock_install_tfp_integration():
        pass
    
    # Replace the real function with our mock
    if hasattr(check_shapes, "integration") and hasattr(check_shapes.integration, "tfp"):
        check_shapes.integration.tfp.install_tfp_integration = mock_install_tfp_integration
        
    # Also patch the initialization to avoid calling the function
    original_init = check_shapes.__init__
    def patched_init():
        # Get all attributes from the original __init__ except install_tfp_integration
        for name in dir(original_init):
            if name != "install_tfp_integration" and not name.startswith("__"):
                setattr(patched_init, name, getattr(original_init, name))
    check_shapes.__init__ = patched_init

import check_shapes
