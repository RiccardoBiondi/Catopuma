'''
Module for the implementation of the loss base classes.
It contains the base abstract class BaseLoss and the implementation of the basic aritmetic operations.
It is possible to sum, subtract, multiply, divide and power two losses or a loss and a float.
If a custom loss is implemented, it must inherit from BaseLoss and implement the __call__ method.
'''

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union, TypeVar


Self = TypeVar("Self", bound="BaseLoss")


class BaseLoss(ABC):
    '''
    Base and abstract class to manage the different loss implementation.
    Each loss must inherit from this class and implement the __call__ method.
    '''
    def __init__(self, name: str = None) -> None:
        
        if name is not None:
            self._name = name
        else:
            self._name = self.__class__.__name__

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Abstract method to implement the loss computation.

        Parameters
        ----------
        y_true: np.ndarray
            ground truth image. It must be a binary image.
        y_pred: np.ndarray
            predicted image. It must be a binary image.
        '''
        raise NotImplementedError("losses must implement a __call__ method")
    
    @property
    def __name__(self) -> str:
        '''
        Loss name.
        '''
        return self._name
        

    @property
    def name(self) -> str:
        '''
        Loss name.
        '''
        return self.__name__
    
    @__name__.setter
    def __name__(self, t_name: str) -> None:
        '''
        Setter for the loss name.
        '''
        self._name = t_name

    @name.setter
    def name(self, t_name: str) -> None:
        '''
        Setter for the loss name.
        '''
        self._name = t_name


    #
    # Define some basic operations like multiplication, 
    # division, addiction, subtraction and power
    #
    
    def __add__(self, other: Union[Self, float]) -> Self:
        '''
        Method to sum two losses or a loss and a float.

        Parameters
        ----------
        other: Union[Self, float]
            second loss to sum or constant to sum. It can be either a loss or a float.
        
        Returns
        --------
        Self
            the sum of the two losses.
        '''

        return LossSum(self, other)
 

    def __radd__(self, other: Union[Self, float]) -> Self:
        '''
        Method to sum two losses or a loss and a float.

        Parameters
        ----------
        other: Union[Self, float]
            second loss to sum or constant to sum. It can be either a loss or a float.
        
        Returns
        --------
        Self
            the sum of the two losses.
        '''
        
        return LossSum(self, other)

    def __sub__(self, other: Union[Self, float]) -> Self:
        '''
        Method to subtract two losses or a loss and a float.

        Parameters
        ----------
        other: Union[Self, float]
            second loss to subtract or constant to subtract. It can be either a loss or a float.
        
        Returns
        --------
        Self
            the subtraction of the two losses.
        '''
        
        return LossSubtract(self, other)

    def __rsub__(self, other: Union[Self, float]) -> Self:
        '''
        '''
        
        return LossSubtract(self, other)

    def __mul__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossMultiply(self, other)

    def __rmul__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossMultiply(self, other)

    def __truediv__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossDivide(self, other)

    def __rtruediv__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossDivide(self, other)

    def __pow__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossPow(self, other)

    def __rpow__(self, other: Union[Self, float]) -> Self:
        '''
        '''

        return LossPow(self, other)
    
    def __neg__(self) -> Self:
        '''
        '''
        return LossMultiply(self, -1.)


class LossSum(BaseLoss):
    '''
    '''

    def __init__(self, loss_1: BaseLoss, loss_2: Union[BaseLoss, float]):

        if isinstance(loss_2, BaseLoss):

            name =  f'{loss_1.__name__}_plus_{loss_2.__name__}'
        elif isinstance(loss_2, float):

            name = f'{loss_1.__name__}_plus_{loss_2}'
        else:
            raise ValueError()
        
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2



    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        '''

        if isinstance(self.loss_2, BaseLoss):
            return self.loss_1(y_true=y_true, y_pred=y_pred) + self.loss_2(y_true=y_true, y_pred=y_pred)
        
        return self.loss_1(y_true=y_true, y_pred=y_pred) + self.loss_2


class LossSubtract(BaseLoss):
    '''
    '''

    def __init__(self, loss_1: BaseLoss, loss_2: Union[BaseLoss, float]):

        if isinstance(loss_2, BaseLoss):

            name =  f'{loss_1.__name__}_minus_{loss_2.__name__}'
        elif isinstance(loss_2, float):

            name = f'{loss_1.__name__}_minus_{loss_2}'
        else:
            raise ValueError()
        
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2



    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        '''

        if isinstance(self.loss_2, BaseLoss):
            return self.loss_1(y_true=y_true, y_pred=y_pred) - self.loss_2(y_true=y_true, y_pred=y_pred)
        
        return self.loss_1(y_true=y_true, y_pred=y_pred) - self.loss_2


class LossMultiply(BaseLoss):
    '''
    '''

    def __init__(self, loss_1: BaseLoss, loss_2: Union[BaseLoss, float]):
        
        if isinstance(loss_2, BaseLoss):
            name = f'({loss_1.__name__})_times_({loss_2.__name__})'
        elif isinstance(loss_2, float):
            name = f'{loss_2}_times_{loss_1.__name__}'
        else:
            raise ValueError()
        
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2


    def __call__(self, y_true, y_pred) -> float:
        
        if isinstance(self.loss_2, BaseLoss):
            return self.loss_1(y_true=y_true, y_pred=y_pred) * self.loss_2(y_true=y_true, y_pred=y_pred)
        return self.loss_1(y_true=y_true, y_pred=y_pred) * self.loss_2

class LossDivide(BaseLoss):
    '''
    '''

    def __init__(self, loss_1: BaseLoss, loss_2: Union[BaseLoss, float]):
        
        if isinstance(loss_2, BaseLoss):
            name = f'({loss_1.__name__})_divided_by_({loss_2.__name__})'
        elif isinstance(loss_2, float):
            name = f'({loss_1.__name__})_divided_by_{loss_2}'
        else:
            raise ValueError()
        
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2

    def __call__(self, y_true, y_pred) -> float:
        
        if isinstance(self.loss_2, BaseLoss):

            return self.loss_1(y_true=y_true, y_pred=y_pred) / self.loss_2(y_true=y_true, y_pred=y_pred)
        
        return self.loss_1(y_true=y_true, y_pred=y_pred) / self.loss_2


class LossPow(BaseLoss):
    '''
    BaseLoss specialization to perform the power between two losses or a loss and a float.
    
    Parameters
    ----------
    loss_1: BaseLoss
        
    loss_2: Union[BaseLoss, float]
        second loss to sum or constant to sum. It can be either a loss or a float.
    '''

    def __init__(self, loss_1: BaseLoss, loss_2: Union[BaseLoss, float]):
        
        if isinstance(loss_2, BaseLoss):
            name = f'({loss_1.__name__})^({loss_2.__name__})'
        elif isinstance(loss_2, float):
            name = f'({loss_1.__name__})^{loss_2}'
        else:
            raise ValueError()
        super().__init__(name=name)
        self.loss_1 = loss_1
        self.loss_2 = loss_2

    def __call__(self, y_true, y_pred):
        '''
        Compute the sum between the two losses.

        '''
        if isinstance(self.loss_2, BaseLoss):

            return self.loss_1(y_true=y_true, y_pred=y_pred) ** self.loss_2(y_true=y_true, y_pred=y_pred)
        
        return self.loss_1(y_true=y_true, y_pred=y_pred) ** self.loss_2