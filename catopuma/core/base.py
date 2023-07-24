import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union, TypeVar


class UploaderBase(ABC):
    '''
    Base class for the vatìrious image rerder.
    It's child calsses must hendling the reading of the images and labels.
    '''

    @abstractmethod
    def __call__(self, *path: Tuple[str]) -> np.ndarray:
        '''
        This is a function that must becarefully implemented. 
        It took as arguments all the path lists, then read them 
        and organize them into the input and target arrays
        '''
        raise NotImplementedError()


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