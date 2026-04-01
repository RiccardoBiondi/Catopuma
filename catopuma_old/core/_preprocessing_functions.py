import os
import numpy as np

from typing import Optional, List
from typing import Union, Callable, Dict, Tuple, List, Optional

from catopuma.core.__framework import _DATA_FORMAT

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


def unpack_labels(image: np.ndarray, labels: Union[int, List[int]], data_format: str = _DATA_FORMAT, **kwargs) -> np.ndarray:
    '''
    Given a single channel labelmap image, unpacks labels associated with image data into separate channels.

    Parameters
    ----------
        image: np.ndarray
            The input labelmap image.
        labels: int or List of integer
            The label or list of labels to unpack.
        data_format: str
            The format of the image data. Default is the data format of the chosen frameworj
    Returns
    -------
        image: np.ndarray:
        An array with unpacked labels in separate channels.

    Example:
        >>> image = np.random.randint(0, 255, size=(10, 64, 64))  # Example image data
        >>> labels = [1, 2, 3]  # Example labels
        >>> unpacked = unpack_labels(image, labels)  # Unpack labels
    '''

    CHANNEL_AXIS: Dict[str, int] = {
        'channels_last': -1,
        'channels_first': 1}
    TRANSPOSE_AXES: Dict[int, List[int]] = {
        4 : [0, 3, 1, 2],
        5: [0, 4, 1, 2, 3]}

    t_labels = [labels] if isinstance(labels, int) else labels
        
    # get the shape by removing the channel dimension.
    # the resulting shape will be (batch, h, w) or (batch, h, w, d)
    # for 2D or 3D case respectively. 
    shape = np.squeeze(image, axis=CHANNEL_AXIS[data_format]).shape
    y = image.reshape(shape)
    # now addthe channel at the end of the shape
    # and create the zero image that will be filled with each different label
    shape = (*shape, len(t_labels))
    y_mc = np.zeros(shape)

    # and finally fill each channel with the corresponding label    
    for i, label in enumerate(t_labels):
        y_mc[..., i] = (y == label).astype(np.uint8)

    if data_format == 'channels_first':
        y_mc = y_mc.transpose(TRANSPOSE_AXES[len(y_mc.shape)])

    return y_mc        
