"""
Functions Module

This module implement some useful function that can be used inside the psecialized pre-processing, data augmentation and losses classes.

"""

import os
import logging
import numpy as np

from typing import Optional, Tuple, List, Dict
from enum import Enum

import tensorflow as tf
import tensorflow.keras.backend as K

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = ['_get_required_axis']


# Some consant definition

ALLOWED_DATA_FORMATS : Tuple[str] = ('channels_first', 'channels_last')

BASE_DATA_FORMAT_REDUCTION_AXIS: Dict[str, List[int]] =   {
                                                                'channels_first': [1, 2],
                                                                'channels_last': [2, 3]
                                                                }



def _get_required_axis(per_image: bool = False, per_channel: bool = False) -> Optional[Tuple[int]]:
    '''
    Funtion to return the collapse axis from the given options
    Parameters
    ----------
    per_image: bool
        Specify if the process will run image-wise or on all the batch
    per_channel: bool
        Specify if the process will run image-wise or on all the channels
    '''

    if per_image is False and per_channel is False:
        return None
    elif per_image is True and per_channel is False:
        return (1, 2, 3)
    elif per_image is False and per_channel is True:
        return (0, 1, 2)
    elif per_image is True and per_channel is True:
        return (1, 2)
    

def _get_roi(image: np.ndarray, thr: Optional[float] = None):
    '''
    Create a ROI by thresholding tha image above the given value (if specified).
    Otherwise the whole image will be considered as ROI

    Parameters
    ----------
    image: np.ndarray
        image from which retrieve the ROI
    thr: float (default None)
        value to threshold above the image
    
    Return
        roi: bool or nop.ndarray[bool]
            mask to use as ROI
    '''

    if thr is not None:
        roi = image > thr
        return ~roi

    return False


def _mask(image: np.ndarray, roi: Optional[np.ndarray] = None, clip: float = 0.):
    '''
    Set all the values outside the rois to clip.
    If the roi is not provided, return the image as it is

    Parameters
    ----------
    image: np.ndarray
        image to mask
    roi: np.ndarray[bool], deafult None
            roi to mask the image with
    clip: float, default 0
        value  to se the value outside the ROI
    '''

    if clip is not None:

        image[roi] = clip
        return image
    return image



def _gather_channels(x: np.ndarray, indexes: Tuple[int], data_format: str = 'channel_last'): # TODO substitute the dataformat string with an Enum
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

    # TODO improve code efficiency

    # Retrieve all the indexes
    if indexes is None:
        return indexes

    if data_format == 'channels_last':
        x = K.permute_dimensions(x, (3, 0, 1, 2))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 2, 3, 0))

    elif data_format == 'channels_first':
        x = K.permute_dimensions(x, (1, 0, 2, 3))
        x = K.gather(x, indexes)
        x = K.permute_dimensions(x, (1, 0, 2, 3))
    
    else:
        raise ValueError(f'Data format: {data_format} is not recognised as valid specification. Allowed dataformat are "channels_last", "channels_first"')

    return x   


def get_reduce_axes(per_image: bool = False, data_format: str = 'channels_last') -> List[int]:
    '''
    '''

    axes = [1, 2] if data_format == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(xs, indexes: Optional[Tuple[int]] = None, data_format: str = 'channels_last') -> np.ndarray:
    '''
    '''
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]

    return [_gather_channels(x, indexes=indexes, data_format=data_format) for x in xs]


def average(x: np.ndarray, per_image: bool = False, class_weights: Optional[np.array] = None, **kwargs) -> float:
    '''
    Average the array 
    '''
    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


