"""
Functions Module

This module implement some useful function that can be used inside the secialized pre-processing, data augmentation and losses classes.


"""

import os
import logging
import numpy as np

from typing import Optional, Tuple, List, Dict
from enum import Enum

import catopuma
# TODO Remove this direct framework dependency
import tensorflow as tf
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = ['gather_channels', 'average', 'get_reduce_axes']


# Some consant definition

ALLOWED_DATA_FORMATS : Tuple[str] = ('channels_first', 'channels_last')

BASE_DATA_FORMAT_REDUCTION_AXIS: Dict[str, List[int]] =   {
                                                                'channels_first': [1, 2],
                                                                'channels_last': [2, 3]
                                                                }

BASE_DATA_FORMAT_GATHING_AXIS: Dict[str, List[int]] = {
                                                        'channels_first': 1,
                                                        'channels_last': -1
                                                        }

def _gather_channels(x: np.ndarray, indexes: Tuple[int], data_format: str = 'channel_last'):
    # TODO substitute the dataformat string with an Enum
    '''
    Retrieves the elements of indices indices in the tensor x.

    Parameters
    ----------
    x: tensor
        tensor from gathe index with
    indexes: Tuple[int]
        tuple specifying the index to gathe
    data_format: str
        either 'channel_last' or 'channel_first'.
        Specify the if the provided data in (batch_size, height, width, channel) or ( batch_size, channel, height, width) format 

    Return
    ------
    x: tensor
        gathered tensor. it is of the same type of the input ones

    Warning
    -------
    This function is implemented to work only with 2D, multichannel, images
    '''

    # Retrieve all the indexes
    if indexes is None:
        return indexes

    if data_format not in BASE_DATA_FORMAT_GATHING_AXIS.keys():
        raise ValueError(f'Data format: {data_format} is not recognised as valid specification. Allowed dataformat are {BASE_DATA_FORMAT_GATHING_AXIS.keys()}')
    
    # TODO here it is dependend to tensorflow, I have to make it compatible also to pytorch
    x = tf.gather(x, indexes, axis=BASE_DATA_FORMAT_GATHING_AXIS[data_format])        

    return x   


def get_reduce_axes(tensor_dims: int = 4, per_image: bool = False, per_channel: bool = False, data_format: str = 'channels_last') -> List[int]:
    '''
    Return the axes to use to reduce the tensor according to the given specification.
    If both per_image and per_channel are provided, then 

    Parameters
    ----------
    tensor_dims: int (default 4)
        dimensions of the tensor to reduce. Usually the dimension is 4, e.g. (batch_size, width, height, channels)
        or 5, e.g. (batch_size, width, height, depth, channels)
    '''

    if data_format not in BASE_DATA_FORMAT_GATHING_AXIS.keys():
        raise ValueError(f'Data format: {data_format} is not recognised as valid specification. Allowed dataformat are {BASE_DATA_FORMAT_GATHING_AXIS.keys()}')

    reduction_axis = list(np.arange(0, tensor_dims))
    
    if per_image:
        reduction_axis.remove(0)
        
    if per_channel:
        
        to_remove = 1 if data_format == 'channels_first' else max(reduction_axis)
        reduction_axis.remove(to_remove)
    
    return reduction_axis


def gather_channels(xs, indexes: Optional[Tuple[int]] = None, data_format: str = 'channels_last') -> np.ndarray:
    '''
    Gather the channels of the tensor according to the given specification.

    Parameters
    ----------
    xs: List[tensor]
        list of tensors to gather
    indexes: Tuple[int]
        tuple specifying the index to gathe
    data_format: str
        either 'channel_last' or 'channel_first'.
        Specify the if the provided data in (batch_size, height, width, channel) or ( batch_size, channel, height, width) format.

    Return
    ------
    xs: List[tensor]
        gathered tensors. Each tensor is of the same type of the input ones
    '''
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]

    return [_gather_channels(x, indexes=indexes, data_format=data_format) for x in xs]


def average(x: np.ndarray, per_image: bool = False, per_channel: bool = False,
            class_weights: Optional[np.array] = None) -> float:
    '''
    Average the tensor to obtain a single value. The averaging is done according to the given specifications.
    Moreover, if the class_weights are provided, the tensor is weighted before averaging.

    Parameters
    ----------
    x: np.ndarray
        tensor to average
    per_image: bool (default False)
        specify if the averaging will be done image-wise or on the whole batch
    per_channel: bool (default False)
        specify if the averaging will be done channel-wise or on the whole image 
    class_weights: np.ndarray
        weights to apply to the tensor before averaging
    data_format: str (default 'channels_last)
        either 'channel_last' or 'channel_first'.
        Specify the if the provided data in (batch_size, height, width, channel) or ( batch_size, channel, height, width) format.

    Return
    ------


    '''

    if per_image:
        x = K.mean(x, axis=0)

    if class_weights is not None:
        x = x * class_weights
    
    if per_channel:
        x = K.mean(x, axis=0)
     
    return K.mean(x)
