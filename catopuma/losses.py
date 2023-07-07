import numpy as np

from catopuma.core.abstraction import BaseLoss
from catopuma.core._loss_functions import f_score

__author__ = ['Riccardo Biondi']





class DiceLosss(BaseLoss):
    '''
    '''

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
        '''
        '''
        return 1 - f_score(y_true=y_true, y_pred=y_pred)