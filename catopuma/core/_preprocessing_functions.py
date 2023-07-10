import os
import numpy as np

from typing import Optional, List


def standard_scale(image: np.ndarray, axis: Optional[List[int]] = None) -> np.ndarray:

    '''
    Stardandize the image according to mean and standard deviation.
    If a roi is specified, the normalization is done computing the mean and the 
    standard deviation inside the provided roi.

    Parameters
    ----------
    image: np.ndarray
        array to standardize
    axis: List[int] (default None)
        axis where compute the mean and std dev

    Return
    ------
    standardize: np.ndarray
        image standardized according to mean ans standard deviation.
    '''
    
    # create the masked array

    mu = np.mean(image, axis=axis, keepdims=True)
    norm = 1. / np.std(image, axis=axis, keepdims=True)

    return norm * (image - mu)


def robust_scale(image: np.ndarray, roi: Optional[np.ndarray] = None, axis: Optional[List[int]] = None) -> np.ndarray:
    pass


def rescale(image: np.ndarray, axis: Optional[List[int]] = None) -> np.ndarray:
    '''
    Rescale the image intensities in [0., 1.]

    image: np.ndarray
        array to rescale
    axis: List[int] (default None)
        axis where rescale

    Return
    ------
    resaled: np.ndarray
        image standardized according to mean ans standard deviation.

    '''
    
    max_ = image.max(axis=axis, keepdims=True)
    min_ = image.min(axis=axis, keepdims=True)

    norm = 1. / (max_ - min_)
    
    return (image - min_) * norm


def identity(image: np.ndarray, **kwargs) -> np.ndarray:
    '''
    Identity transform. Return the image as it is

    Parameter
    ---------
    image: np.ndarray
        input array to return
    
    Return
    ------
    image: no.ndarray
        sme image as input

    '''

    return image