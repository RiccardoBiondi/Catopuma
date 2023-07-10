'''
Module containing the implementation of the vaious Uploader to read and organize the input image
and tergetes.
'''

import os
from typing import Tuple
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from catopuma.core.abstraction import UploaderBase


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

class SimpleITKUploader(UploaderBase):
    '''
    Class to upload and 
    '''

    def __init__(self, data_format: str = 'channels_last') -> None:

        super().__init__()
        if data_format in ['channels_first', 'channels_last']:
            self.data_format = data_format
        else:
            raise ValueError(f'{data_format} is not allowed')

    def __call__(self, *path: Tuple[str]) -> Tuple[np.ndarray]:
        '''
        Call method
        '''

        img = sitk.ReadImage(path[0]) # input path must be first
        tar = sitk.ReadImage(path[1]) # target path must be second

        # convert the images to array
        img = sitk.GetArrayViewFromImage(img)[..., np.newaxis]
        tar = sitk.GetArrayFromImage(tar)[..., np.newaxis]

        if self.data_format == 'channels_first':
            img = img.transpose(2, 0, 1)
            tar = tar.transpose(2, 0, 1)
        return img, tar


class Patch2DLazyUploader(UploaderBase):
    '''
    '''

    def __init__(self, patch_size: Tuple[int] = (16, 16), object_fraction: float = -1.,  data_format: str = 'channels_last'):

        super().__init__()
        if data_format in ['channels_first', 'channels_last']:
            self.data_format = data_format
        else:
            raise ValueError(f'{data_format} is not allowed')
        
        self.patch_size = patch_size
        self.object_fraction = object_fraction
    

    def __call__(self, *path: Tuple[str]) -> Tuple[np.asarray]:
        '''
        '''
        pass