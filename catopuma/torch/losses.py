"""
"""


import numpy as np
from typing import Union, Optional, List

import catopuma
from catopuma.core.base_losses import BaseLoss
from catopuma.core._score_functions import f_score
from catopuma.core._score_functions import tversky_score

from catopuma.core.__framework import _FRAMEWORK as F

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


class DiceLoss(BaseLoss, F.nn.Module):
    """
    """

    def __init__(self, smooth: float = 1e-5, per_image: bool = False, per_channel: bool = False, class_weights: Union[float, List[float]] = 1.,
                class_indexes: Optional[List[int]] = None, data_format: str = 'channels_last', name: Optional[str] = None) -> None:
        
        super().__init__(name=name)
        
        self.smooth = smooth
        self.per_image = per_image
        self.per_channel = per_channel
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.data_format = data_format


    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Convinience method implemented to mantain the compatibility with the multiframework behaviour of
        catopuma. It will call the forward method and return its result.
        '''

        return self.forward(y_true=y_true, y_pred=y_pred)


    def forward(self, y_true, y_pred):
        return 1. - f_score(y_true=y_true, y_pred=y_pred, beta=1., smooth=self.smooth,
                           indexes=self.class_indexes, class_weights=self.class_weights,
                           data_format=self.data_format, per_image=self.per_image, per_channel=self.per_channel)
    


class TverskyLoss(BaseLoss, F.nn.Module):
    '''
    Class to compute the TverskyLoss. The TverskyLoss is internally defined as 1 - tversky_score with alpha=0.5 and beta=0.5.
    The TverskyLoss is a generalization of the DiceLoss, where the alpha and beta parameters can be used to weight the
    precision and recall in the loss computation.

    Parameters
    ----------
    smooth: float (default 1e-5)
        smoothing factor to avoid zero division error and NaN's. Usually is very small and close to the epsilon.
    alpha: float (default 0.5)
        weight to assign to the precision in the loss computation.
    beta: float (default 0.5)
        weight to assign to the recall in the loss computation.
    per_image: bool (dafault False)
        if True, compute the TverskyLoss loss for each image in the batch and then takes the average.
        Otherwise the metric is computed on the whole batch.
    per_channel: bool (default False)
        if True, compute the TverskyLoss loss for each channel in the image and then takes the average.
        Otherwise the metric is computed on the whole image.
    class_weights: List[float] (default 1.)
        weight to assigne to each output channel in the loss computation. If a single value is provided, it will be
        broadcasted to all the channels. The length of the list must be equal to the number of channels.
    class_idexes: List[int], (default None)
        list of inte, specify which channel use in the loss computation. If None, all the channels will be used
    data_format: str, (default 'channels_last').
        Specify the image data format. Can be either channles_last ((btach_size, w, h, channels)) or 'channels_first'
        ((batch_size, channels, w. h)).
    name: str, (default None)
        loss instance name. As default is TverskyLoss
    '''

    def __init__(self, smooth: float = 1e-5, alpha: float = .5, beta: float = .5, per_image: bool = False, per_channel: bool = False,
                 class_weights: Union[float, List[float]] = 1., class_indexes: Optional[List[int]] = None, data_format: str = 'channels_last',
                 name: Optional[str] = None):
        super().__init__(name=name)

        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.per_image = per_image
        self.per_channel = per_channel
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.data_format = data_format

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Compute the TverskyLoss between the ground truth and the predicted image.

        Parameters
        ----------
        y_true: tensor
            ground truth image. Usually a binary image with the same shape of the predicted image.
        y_pred: tensor
            predicted image. Usually a binary image with the same shape of the ground truth image.
        
        Returns
        -------
        float
            TverskyLoss value
        '''

        return 1. - tversky_score(y_true=y_true, y_pred=y_pred, alpha=self.alpha, beta=self.beta, smooth=self.smooth,
                           indexes=self.class_indexes, class_weights=self.class_weights,
                           data_format=self.data_format, per_image=self.per_image, per_channel=self.per_channel)

    def __call__(self, y_true, y_pred):
        '''
        Convinience method implemented to mantain the compatibility with the multiframework behaviour of
        catopuma. It will call the forward method and return its result.
        '''
        return self.forward(y_true=y_true, y_pred=y_pred)