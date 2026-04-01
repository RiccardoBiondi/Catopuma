from typing import Optional
from .base import BackendBase

# This hidden variable will store the unique instance
_BACKEND_INSTANCE: Optional[BackendBase] = None

def get_backend(backend_name: Optional[str] = None) -> BackendBase:
    """
    Singleton Factory for the Catopuma Backend.
    
    If an instance already exists, it returns it. Otherwise, it 
    initializes the requested backend.
    
    Args:
        backend_name: Name of the backend ('numpy', 'torch', 'tf'). 
                     Required for the first call.
    
    Returns:
        The active BackendBase instance.
    """
    global _BACKEND_INSTANCE

    # 1. Return existing instance if available
    if _BACKEND_INSTANCE is not None:
        if backend_name and _BACKEND_INSTANCE.name != backend_name.lower():
            print(f"Warning: Backend '{_BACKEND_INSTANCE.name}' is already active. "
                  f"Ignoring request for '{backend_name}'.")
        return _BACKEND_INSTANCE

    # 2. Ensure a name is provided for the first initialization
    if backend_name is None:
        raise ValueError("Backend not initialized. Please provide a backend_name (e.g., 'torch').")

    # 3. Standard Factory Logic (Lazy Imports)
    name = backend_name.lower().strip()

    if name == "numpy":
        from .np_impl import NumpyBackend
        _BACKEND_INSTANCE = NumpyBackend()

    elif name in ["torch", "pytorch"]:
        try:
            from .torch_impl import TorchBackend
            _BACKEND_INSTANCE = TorchBackend()
        except ImportError:
            raise ImportError("PyTorch not found.")

    elif name in ["tf", "tensorflow"]:
        try:
            from .tf_impl import TFBackend
            _BACKEND_INSTANCE = TFBackend()
        except ImportError:
            raise ImportError("TensorFlow not found.")

    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    return _BACKEND_INSTANCE

def clear_backend():
    """Reset the singleton instance (useful for unit testing)."""
    global _BACKEND_INSTANCE
    _BACKEND_INSTANCE = None