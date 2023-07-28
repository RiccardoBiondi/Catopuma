"""
This module contains the functional implementation of the various losses implemented in this project.
This functions aims to be called by the corresponding losses class and must only do the loss calculation.
The function must take case also of other operations like the channel gating, avere the results per image or on the whole batch or weight the results. 
Each function must takes as arguments the prediction and the ground truth, togheter to the optional arguments to handle the function behaviour behaviour,
and return a floating point value, corresponding to the metric value. 
"""

import numpy as np
import tensorflow as tf
import catopuma
from catopuma.core.framework import _FRAMEWORK_BACKEND as K
from typing import Optional, List, Tuple, Dict, NoReturn, Union

from catopuma.core.functions import get_reduce_axes, average, gather_channels

# TODO: here all the functions works with the tensorflow.keras backend. This could lead to some incompatibilities with the torch API.
# son find a way to solve this and to not assume the backwend

def f_score(y_true, y_pred, beta: float = 1., smooth: float = 1e-5, class_weights: Union[List[float], float] = 1.,  indexes: List[int] = None, per_image: bool = False, data_format: str = 'channels_last'):
    '''
    Function to compute the f1 score between y_true and y_pred using the keras backend.
    The computation includes also a smoothing to avoid Nan's and Inf's.
    During the computation, this function consider only the 1 values for each channel. 
    In this way , if a single channel image containing only 0, 1 is provided, the metrics reflect only the
    performances on the class labelled as 1.

    Parameters
    ----------

    y_true:
        ground truth, used as reference to compute the f score. 
        The shape is (batch, height, width, channels) if  data_format is 'channel_last' (default value) or
        (batch, channels,  height, width) if data format is 'channel_first'.

    y_pred: 
        prediction sample used to compute the f score. Must be a floating point tensor with values in [0., 1.].
        The shape is (batch, height, width, channels) if  data_format is 'channel_last' (default value) or
        (batch, channels,  height, width) if data format is 'channel_first'.

    beta: flaot (default 1.)
        positive real factor chosen such that recall is considered beta times as important as precision.
        if beta=1, the f_score correspond to the Dice - Sorensen Coefficient
    
    smooth: float (default 1e-5)
        smoothing value to use in the loss calculation to avoid division by zero.

    class_weights: float or List of float
        Contribution of each channel to the final score. Can be a single float or a list of float with the same len of the 
        number of considered channels.
    
    indexes: List[int] deafult None
        if provided, the list contaning the index of the channel to consider in the calculation of the loss.
        If a channel index is repeated more than once, the loss is calsulated on the index as many time as the index
        is repeated.
        As default all the channels are considered in the order in which they are in y_true and y_pred.
    
    per_image bool (default false)
        If true, the loss is calculated for each image and then averaged.

    data_format: str (default 'channels_last')
        specitfy the data format of the input samples.
        Can be either 'channels_last', implying a format of (batch, height, width, channels);
        or 'channels_first', implying a format of (batch, channels, height, width)

    
    Return
    ------
    score: flaot
        f score resulting form the computation. It is in [0, 1]
    '''
    gt = gather_channels(y_true, indexes=indexes, data_format=data_format)
    pr = gather_channels(y_pred, indexes=indexes, data_format=data_format)

    # clip the values to avoid NaN's and Inf's
    axes = get_reduce_axes(per_image=per_image, data_format=data_format)

    # calculate score
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, np.asarray(class_weights))

    return score


def tversky_score(y_true, y_pred, alpha: float = .5, beta: float = .5, smooth: float = 1e-5, class_weights: Union[List[float], float] = 1., 
                  indexes: List[int] = None, per_image: bool = False, data_format: str = 'channels_last'):
    '''
    Function to compute the twersky metric . It will be a floating point value in [0., 1.].
    This function allows to chose the conribution of precision and recall in loss computation, 
    by setting the parameters alpha and beta. 
    If alpha = .5 and beta = .5, it is the dice coefficient. If alpha = beta = 1, it is the 
    Jaccard index.

    Parameters
    ----------
    y_true:
        ground truth, used as reference to compute thetversky score. 
        The shape is (batch, height, width, channels) if  data_format is 'channel_last' (default value) or
        (batch, channels,  height, width) if data format is 'channel_first'.

    y_pred: 
        prediction sample used to compute the thetversky score. Must be a floating point tensor with values in [0., 1.].
        The shape is (batch, height, width, channels) if  data_format is 'channel_last' (default value) or
        (batch, channels,  height, width) if data format is 'channel_first'.

    alpha: 

    beta:

    smooth: float (default 1e-5)
        smoothing value to use in the loss calculation to avoid division by zero.

    class_weights: float or List of float
        Contribution of each channel to the final score. Can be a single float or a list of float with the same len of the 
        number of considered channels.
    
    indexes: List[int] deafult None
        if provided, the list contaning the index of the channel to consider in the calculation of the loss.
        If a channel index is repeated more than once, the loss is calsulated on the index as many time as the index
        is repeated.
        As default all the channels are considered in the order in which they are in y_true and y_pred.
    
    per_image bool (default false)
        If true, the loss is calculated for each image and then averaged.

    data_format: str (default 'channels_last')
        specitfy the data format of the input samples.
        Can be either 'channels_last', implying a format of (batch, height, width, channels);
        or 'channels_first', implying a format of (batch, channels, height, width)

    Return
    ------
    score: flaot
        tversy score resulting form the computation. It is in [0, 1]
    '''

    gt = gather_channels(y_true, indexes=indexes, data_format=data_format)
    pr = gather_channels(y_pred, indexes=indexes, data_format=data_format)

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # clip the values to avoid NaN's and Inf's
    axes = get_reduce_axes(per_image=per_image, data_format=data_format)

    # calculate score
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    score = average(score, per_image, np.asarray(class_weights))

    return score