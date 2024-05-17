"""
This module implements some useful functions to perform padding and unpadding operations.
Up to now these functions are mainly used by the PatchPredict class to perform a valid patch 
prediction.

The implemented methods allows to determine the padding values and pad the specified tensors.

The implemented functions are an adaptation of the one implemented by Gianluca Carlini
in https://github.com/GianlucaCarlini/Segmentation3D/blob/main/utils/window_operations.py 
"""

import numpy as np
from typing import Iterable, Callable, Dict, Tuple, List

from catopuma.core.__framework import _DATA_FORMAT
from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK_BASE as B
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K

# Import the padding funcion according to the framework in use
if _FRAMEWORK_NAME == 'torch':
    from torch.nn.functional import pad as pad_function

    def min_function(X):
        return X.min()

else:
    from tensorflow import pad as pad_function
    from tensorflow import reduce_min as min_function

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['PADDING_GETTERS', 'get_padding_values_for_strides']

# Now define the function to add the channel padding values according to the data format.
# the if statement is executed here to avoid uneccessary repetitions.
# Notice that only one data format must working at time

if _DATA_FORMAT == 'channels_first':
    
    def _add_channel_values(pad_values: List[List[int]], channel_values):
        '''
        '''

        pad_values.insert(0, channel_values)
        return pad_values
else:
    def _add_channel_values(pad_values: List[List[int]], channel_values):
        '''
        '''

        pad_values.append(channel_values)
        return pad_values 

# specify the axis where the channels padding values will be added.
# the axis depends on the data format
_CHANNEL_AXIS: Dict[str, int] = {
    'channels_last': -1,
    'channels_first': 0
}

def _get_valid_padding_values_for_strides(array_shape: Iterable[int], strides: Iterable[int], patch_size: Iterable[int]) -> Iterable[Iterable[int]]:
    '''
    Return the start and stop values along each image dimension to  to make it compatible with the patch size and strides.

    Parameters
    ----------

    array_shape: Iterable[int]
        shape of the array/ tensor to pad. Must contains only height and width for the 2D case or
        height, width and depth for the 3D case.
    strides: Iterable[int]
        Must have the same lenght of array_shape. Strides to used to slide the patch 
        during the patch prediction loop.
    patch_size: Iterable[int]:
        shape of the patch considered for the prediction.

    Return
    ------
    padding_values: Iterable[Iterable[int]]
       Iterable containing starting and stopping index for the resulting image array.

    TODO: Check the formula for the stop index! Could not be correct
    '''
    # this value is basically the number of patches that can be extracted along each direction.
    # References for this computation could be founded at: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # or at https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    depth_values = [np.ceil((d - p + 1) / s).astype(np.uint8) for d, p, s in zip(array_shape, patch_size, strides)]

    # once I have computed the number of patches along each direction, define the starting index
    # and the stop index for each dimensions.
    # The stop index should be lower or equal to the image shape along the specific direction.
    # Up to now the formula I came up with si : strides * depth_values

    padding_values = [[0, s * dv] for dv, s in zip(depth_values, strides)]

    # now add te values to make the shape compatible with the padding 
    padding_values = _add_channel_values(padding_values, [None, None])
    # now add the padding values for the batch axis, corresponding to the zero dimension
    _ = padding_values.insert(0, [None, None])

    return padding_values


def _get_same_padding_values_for_strides(array_shape: Iterable[int], strides: Iterable[int], patch_size: Iterable[int]):
    """
    Get padding values to make it compatible with the patch size and strides.
    The padding is performed in a tensorflow-like fashion.

    Parameters
    ----------
    array_shape: Iterable[int]
        shape of the array/ tensor to pad. Must contains only height and width for the 2D case or
        height, width and depth for the 3D case.
    strides: Iterable[int]
        Must have the same lenght of array_shape. Strides to used to slide the patch 
        during the patch prediction loop.
    patch_size: Iterable[int]:
        shape of the patch considered for the prediction.

    Return
    ------
    pad_values: Iterable[Iterable[int]]

        iterable containing the pair of padding values for each direction. 

    """
    
    # first of all compute the padding depth for each direction.
    # This represents the toal amount of padding to add in each direction
    pad_depth = [max(p - s, 0) if d % s == 0 else max(p - (d % s), 0) for p, s, d in zip(patch_size, strides, array_shape)]

    # now compute the values to add before and after for each direction
    pad_values = [[depth // 2, depth - (depth // 2)] for depth in pad_depth]

    # now add the channel dimension according to the data format
    # and then add the batch dimension
    pad_values = _add_channel_values(pad_values, [0, 0])
    _ = pad_values.insert(0, [0, 0])

    return pad_values


PADDING_GETTERS: Dict[str, Callable] = {
    "valid": _get_valid_padding_values_for_strides,
    "same": _get_same_padding_values_for_strides
}


def get_padding_values_for_strides(array_shape: Iterable[int], strides: Iterable[int], patch_size: Iterable[int], padding: str = "same") -> Iterable[int]:
    """
    Get padding values to make the input tensor it compatible with the patch size and strides.
    The values could be computed for both valid and same (default) padding modes.
    In case of valid, it will return the lower index and size for each dimension that contains exactly the imaged.
    In case of same, it will return the left and right value to append to the tenesor to make the patch prediction return an image of the same shape as the input one.

    Parameteers
    -----------
    array_shape: Iterable[int]
        shape of the input array from which retrieve the padding values. it must be (height, width) for 2D case  or (height, width, depth)
        for the 3D case. No batch or channel information should be provided.
    strides_shape: Iterable[int]
        Strides to used to slide the patch during the patch prediction loop. Must have the same lenght of array_shape.
    patch_size: Iterable[int]:
        shape of the patch considered for the prediction.  Must have the same langht of array_shape

    padding: str (dafault "same")
        padding method to use for the estimation of the padding values. allowed values are 'valid' and 'same' (case sensitive). 

    Returns
    -------

    padding_values: Iterable[Iterable[int]]
        list of padding values estimated according to the specified padding modality

    Raise
    -----

    ValueError: if the specified padding is different from 'valid' or 'same'

    ValuError: if the len of strides or the len of patch_size are different from the len of array_shape
    """
    
    if padding not in PADDING_GETTERS.keys():
        raise ValueError(f"Specified padding modality {padding} not supported. Allowed modalities are {PADDING_GETTERS.keys()}")

    # this check is due to the fact that the function requires array_shape, strids and patch_size
    # to have the same len. Only two check are necessary due to the transitiv property. 
    if (len(array_shape) != len(strides)) | (len(array_shape) != len(patch_size)):
        raise ValueError(f"lenght of array_shape and strieds does not match: {len(array_shape)} != {len(strides)}")

    # call the proper function selected from the dictionary according to the padding key
    return PADDING_GETTERS[padding](array_shape, strides, patch_size)


def _pad_tensor_same(tensor, padding_values: Tuple[Tuple[int]]):
    """
    Pad the tensor in same fashion by adding constant voxels at image border. 
    The constant is the minumum image value. 
    The input padding values must be specified as tensorflow requires.
    They are internally converted in the torch ones if the active framework is torch

    Parameters
    ----------
    tensor: input tensor to pad
        input tensor to pad.
        Its shape must be:
            -  (n_channels, h, w) or (n_channels, h, w, d) if data_format channels first 
            -  (h, w, n_channels) or ( h, w, d, n_channels) if data_format channels last 
    padding_values: Tuple[Tuple[int]]
        tuple with the padding depth in each direction for each dimension, included the channels.
        To achieve a good padding, the padding depth values along channel direction must be [0, 0].

    Return
    ------
    padded tensor
        tensor padded as specified. The type of padding is constant and the constant values is the
        minimum value of the tensor.

    TODO: add a way to choose the kind of padding.
    """
    # if the current framework is torch, convert the padding values to the correct format
    if _FRAMEWORK_NAME == 'torch':
        padding_values = sum(list(map(list, padding_values)), [])[::-1]
    # Using constant in lower case as specification does not introduce framework incompatiility
    # since the function from tensorflow is case-insensitive.
    return pad_function(tensor, padding_values, 'constant', min_function(tensor))


def _pad_tensor_valid(tensor, padding_values: Tuple[Tuple[int]]):
    '''
    Crop the tensor in order to achieve a tensor where the extracte patch fit perfectly.

    Parameters
    ----------
    tensor: tensor
        input tensor to pad, its shape must be:
            -  (n_channels, h, w) or (n_channels, h, w, d) if data_format channels first 
            -  (h, w, n_channels) or ( h, w, d, n_channels) if data_format channels last 
    padding_values: Tuple[Tuple[int]]
        tuple with the padding depth to remove in each direction for each dimension, included the channels.
        To achieve a good padding, the padding depth values along channel direction must be [None, None]
        to include all the channels.
        The other index, resulting from the _get_valid_padding_values_for_strides function, should 
        be the staring and ending index of the slicing.

    Return
    ------
    padded_tensor: tensor
        the tensor with a shape capable to exactly contain the patches.
    '''

    # from the padding_values create the tuple containing the slices.
    # the slices corresponding to the channels will be [None, None] in order to 
    # include all the channels. The correct slicing formatting must be checked BEFORE 
    # passing the padding_values to the function.
    slices = tuple([slice(*s) for s in padding_values])

    return tensor[slices]


PADDING_METHODS: Dict[str, Callable] = {
    "valid": _pad_tensor_valid,
    "same": _pad_tensor_same
}


def pad_tensor(tensor, padding_values: Iterable[Iterable[int]], padding: str ='same'):
    '''
    Pad the specified tensor according to the chosen modality: 'same' or 'valid'.

    Parameters
    ----------
    tensor: 
        input tensor to pad.
    padding_values: Iterable[Iterable[int]]
        matrix containing the padding values for each direction
    padding: str
        either one of 'same' of 'valid'. 
    
    Return
    ------

    padded: 
        the padded tensor
    
    Raise
    -----
    ValueError:
        raise a value error if the specified padding modality is different from 
        'same' or 'valid'
    '''
    if padding not in PADDING_METHODS.keys():
        raise ValueError(f'the specified padding modality: {padding} is not allowd. Supported modalities are: {PADDING_METHODS.keys()}')

    return PADDING_METHODS[padding](tensor, padding_values)