'''
This modules implements some pre-processing classes to be passed to the feeder for the on-the-fly 
preprocessing of the input images
'''

import numpy as np

from typing import Union, Callable, Dict, Tuple, List, Optional

from catopuma.core.base import PreProcessingBase

from catopuma.core._base_functions import get_reduce_axes
from catopuma.core._preprocessing_functions import standard_scaler
from catopuma.core._preprocessing_functions import robust_scaler
from catopuma.core._preprocessing_functions import min_max_scaler
from catopuma.core._preprocessing_functions import identity
from catopuma.core._preprocessing_functions import unpack_labels

from catopuma.core.__framework import _DATA_FORMAT

__author__ = ['Riccardo  Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['PreProcessing']


SCALER_LUT: Dict[str, Callable] = {
                'standard': standard_scaler,
                'robust': robust_scaler,
                'minmax': min_max_scaler,
                'identity': identity,
                'unpack_labels': unpack_labels
                }


class PreProcessing(PreProcessingBase):
    '''
    Class implementing a simple preprocessing. 
    The implemented preprocessing consist of: 
        - image normalization according to the specified scaler and options
        - target label selection (also multiple label are allowed)
    
    Parameters
    ----------
    standardizer: str of Callable. (default 'identity')
        specify the kind of standardization of the image. Available string are: 
            - identity: no normalization is applied
            - standard: normalization according to mean and standard deviation
            - robust: normalization according to median and inter quartile range
            - minmax: rescale the gray level in [min, max] range. Dafault to [0, 1]
            - unpack_labels: Unpacks labels associated with image data into separate channels

        If it is a callable, must takes as first argument the image array. Eventually it can also takes the axis argument, 
        to allow per_image and per_channel standardization. the optional argument data_format allows the correct specification of the 
        reduction axis
    per_image: bool (default False)
        if true, each image is standardized as single entity, otherwise the normalization is performed on the whole batch
    per_channel: bool (default False)
        if true, normalization is performed channel-wise
    data_format: str (default 'channels_last')
        specify if the data are in (batch_size, h, w, channel) shape (channels_last) or in (batch_size, channel, h, w) format (channels_first).
    target_standardizer: str or Callable. (default 'identity')
        Specify the kind of preprocessing of the tharget image.
        Avalable strings are:
            - identity: no normalization is applied
            - standard: normalization according to mean and standard deviation
            - robust: normalization according to median and inter quartile range
            - minmax: rescale the gray level in [min, max] range. Dafault to [0, 1]
            - unpack_labels: Unpacks labels associated with image data into separate channels
    standardizer_params: Dict
        parameters of the standardizer
    target_standardizer_params: Dict
        parameters of the target_standardizer
    '''

    def __init__(self,
                standardizer: Union[str, Callable] = 'identity',
                per_image: bool = False,
                per_channel: bool = False,
                data_format: str = _DATA_FORMAT,
                target_standardizer: Union[str, Callable] = 'identity',
                standardizer_params: Dict = {},
                target_standardizer_params: Dict = {}):
        '''
        '''
        
        if isinstance(standardizer, str) & (standardizer not in [k for k, _ in SCALER_LUT.items()]):
            raise ValueError(f'Unknown standardized method: {standardizer}')
        
        if isinstance(target_standardizer, str) & (target_standardizer not in [k for k, _ in SCALER_LUT.items()]):
            raise ValueError(f'Unknown standardized method: {target_standardizer}')
        
        # TODO implement the data format checking into a decorator
        if data_format not in ('channels_last', 'channels_first'):
            raise ValueError(f'Unknown data format {data_format}. It must be one of "channels_last" or "channels_first"')


        if isinstance(standardizer, str):
            self.standardizer = SCALER_LUT[standardizer]
        else: 
            self.standardizer = standardizer
        
        if isinstance(target_standardizer, str):
            self.target_standardizer = SCALER_LUT[target_standardizer]
        else: 
            self.target_standardizer = target_standardizer


        self.per_image = per_image
        self.per_channel = per_channel
        self.data_format = data_format

        self.standardizer_params = standardizer_params
        self.target_standardizer_params = target_standardizer_params

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        '''
        Perform the specified preprocessing on the batch

        Parameters
        ----------
        X: np.ndarray
            input image to normaize
        y: np.ndarray
            target label

        Return
        ------
        X: np.ndarray
            normalized image
        y: np.ndarray
            target label
        '''

        axis = get_reduce_axes(tensor_dims=len(X.shape), per_image=self.per_image, per_channel=self.per_channel, data_format=self.data_format)

        X = self.standardizer(X, axis=axis, **self.standardizer_params)
        y = self.target_standardizer(y, **self.standardizer_params)

        X = X.astype('float')
        y = y.astype('float')

        return X, y