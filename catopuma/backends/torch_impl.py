import torch
import numpy as np
from .base import BackendBase
from typing import Any

class TorchBackend(BackendBase):

    _DTYPE_MAP = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        # Aggiungiamo il supporto diretto alle classi numpy se passate come tipo
        float: torch.float32,
        int: torch.int64,
    }

    def __init__(self):
        super().__init__(channel_format="channels_first")


    def name(self) -> str:
        return "torch"

    # type cast
    def numpy_to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data)

    def tensor_to_numpy(self, data: torch.Tensor) -> np.ndarray:
        return data.numpy(force=True)

    # math operations
    def sum(self, x: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        return torch.sum(x, dim=axis, keepdims=keepdims)        
        
    def mean(self, x: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        """Compute the arithmetic mean across specified axes."""
        return torch.mean(x, dim=axis, keepdims=keepdims)

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the element-wise absolute value."""
        return torch.abs(x)

    def pow(self, x: torch.Tensor, exponent: float) -> torch.Tensor:
        return torch.pow(x, exponent=exponent)

    def clip(self, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return torch.clip(x, min=min_val, max=max_val)

    def log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


    # slicing method implementation
    def slice(self, x: torch.Tensor, starts: tuple, sizes: tuple) -> torch.Tensor:
        """
        Extracts a patch from a Torch tensor.
        Example: if starts=(10, 20) and sizes=(5, 5), it performs x[10:15, 20:25]
        """
        # Create a tuple of slice objects: slice(start, start + size)
        # This works for any number of dimensions (2D, 3D, etc.)
        slices = tuple(slice(s, s + sz) for s, sz in zip(starts, sizes))
        
        # Applying the tuple of slices to the tensor
        return x[slices]

    def scatter_update(self, canvas: torch.Tensor, patch: torch.Tensor, starts: tuple) -> torch.Tensor:
        """
        Updates a region of the canvas with a patch.
        Note: In PyTorch, this is an in-place operation by default.
        """
        sizes = patch.shape
        slices = tuple(slice(s, s + sz) for s, sz in zip(starts, sizes))
        
        # We use no_grad to ensure this doesn't interfere with the 
        # computation graph if we are just reconstructing for visualization
        with torch.no_grad():
            canvas[slices] = patch  
        return canvas

    def zeros(self, shape: tuple, dtype: Any) -> torch.Tensor:
        """
        Create a tensor of zeros with the given shape.
        Used to initialize accumulators for patch-based reconstruction.
        """
        return torch.zeros(shape, dtype=self._DTYPE_MAP[dtype])