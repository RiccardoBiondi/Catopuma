'''
This modules implements some pre-processing classes to be passed to the feeder for the on-the-fly 
preprocessing of the input images
'''

import numpy as np
import tensorflow as tf

from typing import Union, Callable, Dict, Tuple

from catopuma.core.abstraction import PreProcessingBase

from catopuma.core.functions import _get_required_axis
from catopuma.core._preprocessing_functions import standard_scale
from catopuma.core._preprocessing_functions import rescale
from catopuma.core._preprocessing_functions import identity

__author__ = ['Riccardo  Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

__all__ = ['PreProcessing']


SCALER_LUT: Dict[str, Callable] = {
                'standard_scale': standard_scale,
                'rescale': rescale,
                'identity': identity
                }


class PreProcessing(PreProcessingBase):
    '''
    Class implementing a simple preprocessing. 
    The implemented preprocessing consist of: 
        - image normalization according to the specified scaler and options
        - target label selection
        - conversion of the images from numpy array to tensorflow tensors
    
    Parameters
    ----------
    standardizer: str of Callable. (default 'identity')
        specify the kind of standardization of the image. Available string are: 
            - identity: no normalization is applied
            - standard_scale: normalization according to mean and standard deviation
            - rescale: rescale the gra level in [0, 1]
        If it is a callable, must takes as first argument the image array. Eventually it can also takes the axis argument, 
        to allow per_image and per_channel standardization. the optional argument data_format allows the correct specification of the 
        reduction axis
    per_image: bool (default False)
        if true, each image is standardized as single entity, otherwise the normalization is performed on the whole batch
    per_channel: bool (default False)
        if true, normalization is performed channel-wise
    data_format: str (default 'channels_last')
        specify if the data are in (batch_size, h, w, channel) shape (channels_last) or in (batch_size, channel, h, w) format (channels_first).

    target_label: int (default 1)
        the label to use as a target for the model.
    

    '''

    def __init__(self,  standardizer: Union[str, Callable] = 'identity', per_image: bool = False, per_channel: bool = False, data_fromat: str = 'channels_last', target_label: int = 1):
        
        if isinstance(standardizer, str) & (standardizer not in [k for k, _ in SCALER_LUT.items()]):
            raise ValueError(f'Unknown standardized method: {standardizer}')
        
        if data_fromat not in ('channels_last', 'channels_first'):
            raise ValueError(f'Unknown data format {data_fromat}. It must be one of "channels_last" or "channels_first"')


        if isinstance(standardizer, str):
            self.standardizer = SCALER_LUT[standardizer]
        else: 
            self.standardizer = standardizer

        self.per_image = per_image
        self.per_channel = per_channel
        self.data_format = data_fromat
        self.target_label = target_label

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
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
    
        # normalizza sull'intare bathc: non sulla singola immagine.Aggiungi un per_image
        X = X.astype('float')
        y = y.astype('float')
        y = (y == self.target_label).astype('float')

        axis = _get_required_axis(per_image=self.per_image, per_channel=self.per_channel)

        X = self.standardizer(X, axis=axis)

        return tf.convert_to_tensor(X), tf.convert_to_tensor(y)
