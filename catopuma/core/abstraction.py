import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple



class UploaderBase(ABC):
    pass


class PreProcessingBase(ABC):
    
    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        '''

        raise NotImplementedError()


class DataAgumentationBase(ABC):
    
    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        '''

        raise NotImplementedError()
    


class BaseLoss(ABC):
    def __init__(self, name: str = None) -> None:

        self._name = name

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("losses must implement a __call__ method")
    
    @property
    def __name__(self) -> str:
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    @property
    def name(self) -> str:
        return self.__name__
    
    @__name__.setter
    def __name__(self, t_name: str) -> None:
        self.__name__ = t_name

    @name.setter
    def name(self, t_name: str) -> None:
        self.__name__ = t_name