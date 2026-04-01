from abc import ABC, abstractmethod
from typing import Any, TypeVar

Tensor = TypeVar("Tensor")
NpArray = TypeVar("NpArray")

class BackendBase(ABC):
    """
    Abstract Base Class defining the contract for framework-agnostic operations.
    
    This class serves as the translation layer between Catopuma's core logic 
    (Losses, Patch Engines, Preprocessing) and specific deep learning frameworks 
    like PyTorch, TensorFlow, or JAX.
    """
    # name of the backend
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the backend (e.g., 'pytorch', 'tensorflow', 'numpy')."""
        pass

    # type cast
    @abstractmethod
    def numpy_to_tensor(self, data: NpArray) -> Tensor:
        """
        Convert a NumPy array or array-like object into a framework-specific tensor.
        
        Args:
            data: The input NumPy array.
        Returns:
            A tensor allocated on the default device of the active backend.
        """
        pass

    @abstractmethod
    def tensor_to_numpy(self, data: Tensor) -> NpArray:
        """
        Convert a framework-specific tensor back into a NumPy array.
        
        Args:
            data: The framework-specific tensor.
        Returns:
            A NumPy array on CPU.
        """
        pass

    # math operations

    @abstractmethod
    def sum(self, x: Tensor, axis: int = None, keeepdims: bool = False) -> Tensor:
        """Compute the sum of elements across specified axes."""
        pass

    @abstractmethod
    def mean(self, x: Tensor, axis: int = None, keeepdims: bool = False) -> Tensor:
        """Compute the arithmetic mean across specified axes."""
        pass

    @abstractmethod
    def abs(self, x: Tensor) -> Tensor:
        """Compute the element-wise absolute value."""
        pass

    @abstractmethod
    def pow(self, x: Tensor, exponent: float) -> Tensor:
        """Compute element-wise power (x^exponent)."""
        pass

    @abstractmethod
    def clip(self, x: Tensor, min_val: float, max_val: float) -> Tensor:
        """
        Clip (limit) the values in a tensor between a min and max threshold.
        Commonly used for Hounsfield Unit (HU) windowing in medical imaging.
        """
        pass

    @abstractmethod
    def log(self, x: Tensor) -> Tensor:
        """Compute the natural logarithm element-wise."""
        pass

    @abstractmethod
    def exp(self, x: Tensor) -> Tensor:
        """Compute the exponential (e^x) element-wise."""
        pass


    # slicing method implementation
    @abstractmethod
    def slice(self, x: Tensor, starts: tuple, sizes: tuple) -> Tensor:
        """
        Extract a sub-region (patch) from a tensor.
        
        Args:
            x: The input tensor (volume or image).
            starts: Starting indices for each dimension (e.g., (z, y, x)).
            sizes: Size of the region to extract for each dimension.
        Returns:
            A tensor representing the extracted patch.
        """
        pass

    @abstractmethod
    def scatter_update(self, canvas: Tensor, patch: Tensor, starts: tuple) -> Tensor:
        """
        In-place or functional update of a region within a larger tensor.
        Crucial for reconstructing full volumes from individual predicted patches.
        
        Args:
            canvas: The destination tensor where the patch will be placed.
            patch: The source tensor to insert.
            starts: The coordinates in the canvas where the patch insertion begins.
        Returns:
            The updated canvas tensor.
        """
        pass

    @abstractmethod
    def zeros(self, shape: tuple, dtype: Any) -> Tensor:
        """
        Create a tensor of zeros with the given shape.
        Used to initialize accumulators for patch-based reconstruction.
        """
        pass