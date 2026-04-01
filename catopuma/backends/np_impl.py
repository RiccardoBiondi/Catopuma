import numpy as np
from typing import Any, Tuple
from .base import BackendBase


class NumpyBackend(BackendBase):
    """
    NumPy-based implementation of the Catopuma Backend.
    Provides a non-differentiable reference for data loading and preprocessing.
    """

    def __init__(self):
        super().__init__()

    # backend name
    @property
    def name(self) -> str:
        return "numpy"

    # --- Type Casting & Conversion ---

    def numpy_to_tensor(self, data: np.ndarray) -> np.ndarray:
        # In this backend, the "tensor" is already a numpy array
        return np.asanyarray(data)

    def tensor_to_numpy(self, data: np.ndarray) -> np.ndarray:
        return np.asanyarray(data)

    # --- Math Operations ---

    def sum(self, x: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def pow(self, x: np.ndarray, exponent: float) -> np.ndarray:
        return np.power(x, exponent)

    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(x, a_min=min_val, a_max=max_val)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def exp(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    # --- Slicing & Patch Manipulation ---

    def slice(self, x: np.ndarray, starts: Tuple[int, ...], sizes: Tuple[int, ...]) -> np.ndarray:
        """
        Extracts a patch using standard NumPy slicing: x[s:s+sz, ...]
        """
        slices = tuple(slice(s, s + sz) for s, sz in zip(starts, sizes))
        return x[slices]

    def scatter_update(self, canvas: np.ndarray, patch: np.ndarray, starts: Tuple[int, ...]) -> np.ndarray:
        """
        Updates the canvas in-place. NumPy is mutable, so this is very efficient.
        """
        sizes = patch.shape
        slices = tuple(slice(s, s + d) for s, d in zip(starts, sizes))
        canvas[slices] = patch
        return canvas

    def zeros(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)