import numpy as np

from typing import Union, Optional, List

from catopuma.core.abstraction import BaseLoss
from catopuma.core._loss_functions import f_score

__author__ = ['Riccardo Biondi']


class DiceLoss(BaseLoss):
    '''
    Function to compute the DiceLoss. The dice loss is internally defined as 1 - f_score with beta=1,
    corresponding to the armonic mean of precision and recall, therefore to the dice-sorensen coefficient.

    Parameters
    ----------

    '''

    def __init__(self, smooth: float = 1e-5, per_image: bool = False, class_weights: Union[float, List[float]] = 1.,
                    class_indexes: Optional[List[int]] = None, data_format: str = 'channels_last', name: Optional[str] = None):
        super().__init__(name=name)

        self.smooth = smooth
        self.per_image = per_image
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.data_format = data_format


    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        '''
        return 1 - f_score(y_true=y_true, y_pred=y_pred, beta=1., smooth=self.smooth,
                           class_indexes=self.class_indexes, class_weights=self.class_weights,
                           data_format=self.data_format, per_image=self.per_image)