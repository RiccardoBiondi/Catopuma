import os
import numpy as np

from typing import Optional, List


def standard_scaler(image: np.ndarray, axis: Optional[List[int]] = None) -> np.ndarray:

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


def robust_scaler(image: np.ndarray, roi: Optional[np.ndarray] = None, axis: Optional[List[int]] = None) -> np.ndarray:
    '''
    Rescale the image intensities according to median and interquartile range.
    If a roi is specified, the normalization is done computing the median and the
    interquartile range inside the provided roi.

    Parameters
    ----------
    image: np.ndarray
        array to rescale
    roi: np.ndarray (default None)
        region of interest where compute the median and the interquartile range. 
        Must be binary and have the same shape of the image.
    axis: List[int] (default None)
        reduction axis where compute the median and the interquartile range
    
    Return
    ------

    rescaled: np.ndarray
        image standardized according to median and interquartile range.
        The rescaled image have the same shape as the input image.
    '''

    median = np.median(image, axis=axis, keepdims=True)
    iqr = np.subtract(*np.percentile(image, [75, 25], axis=axis, keepdims=True))
    scale_factor = 1. / iqr
    rescaled = (image - median) * scale_factor
    
    return rescaled


def min_max_scaler(image: np.ndarray, axis: Optional[List[int]] = None) -> np.ndarray:
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