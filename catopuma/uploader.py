'''
Module containing the implementation of the vaious Uploader to read and organize the input image
and tergetes.
'''

import os
from typing import Tuple, NoReturn
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from catopuma.core.base import UploaderBase


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


class LazyPatchBaseUploader(UploaderBase):
    '''
    Loader to peform a patch base lazy loading of medical images. 
    The loader is based on the SimpleITK reader.

    This implementation is inspired by the one of Gianluca Carlini:
    https://github.com/GianlucaCarlini/Segmentation3D/blob/main/loaders/lazy_loaders.py

    Parameters
    ----------
    patch_size: Tuple[int]
        patch size. Must be specified for each image dimensions. 
    threshold: float, (default -1.)
        Threshold value to consider for patch sampling.
        If the sum of non-zero pixels in the sampled patch is lower than
        threshold, then another patch is sampled until the threshold condition is met.
        As default each patch will be taken as valid
    data_format:   str, (default 'channels_last')
        data format of the returned images. Can be either 'channels_last' (i.e. (batch_size, w, h, channels))
        or 'channels_first' (i.e., (batch_size, w, h, channels))

    '''
    
    def __init__(self, patch_size: Tuple[int], threshold: float = -1., data_format: str = 'channels_last') -> None:
        
        super().__init__()
        if data_format in ['channels_first', 'channels_last']:
            self.data_format = data_format
        else:
            raise ValueError(f'{data_format} is not allowed')
            
        self.patch_size = patch_size
        self.threshold = threshold
    
    def __call__(self, *path: Tuple[str]) -> Tuple[np.array]:
        
        reader = sitk.ImageFileReader()
        _ = reader.SetFileName(path[-1])
        _ = reader.ReadImageInformation()
        image_size = reader.GetSize()
        
        _ = self._checkConsistency(image_size)
        
        # insert the threshold case
        condition = True
        
        while condition:
            
            patch_origin = self._samplePatchOrigin(image_size)
            _ = reader.SetExtractIndex(patch_origin)
            _ = reader.SetExtractSize(self.patch_size)
            y = sitk.GetArrayFromImage(reader.Execute())[..., np.newaxis]
            
            if np.sum(y > 0.0) > self.threshold * np.prod(np.asarray(self.patch_size)):
                condition = False
            
        _ = reader.SetFileName(path[0])
        X = sitk.GetArrayFromImage(reader.Execute())[..., np.newaxis]
        
        if self.data_format == 'channels_first':
            X = X.transpose(2, 0, 1)
            y = y.transpose(2, 0, 1)

        return X, y
            
        
        # extract and return (as batch and channel shaped)
    
    def _samplePatchOrigin(self, image_size: Tuple[int]) -> Tuple[int]:
        '''
        Sample the patch origin according to a uniform distribution ensuring 
        that all the patch is inside the the image
        
        Parameter
        ---------
            image_size: Tuple[int]
                size of the image from which extract the patch origin
        Return
        ------
            origin: Tuple[int]
                patch origin
        '''
        origin = [np.random.randint(0, max(i - p, 1)) for i,p in zip(image_size, self.patch_size)]
        
        return origin

    def _checkConsistency(self, image_size: Tuple[int]) -> NoReturn:
        '''
        Check the image from which extract the patch have the same dimension of the 
        specified patch size. Also check that the patch size is lower than the image
        size, i.e. the patch is smaller or equal than the image.
        
        Parameters
        ----------
        image_size: Tuple[int]
            size of the image from which extract the patch
        
        Raise
        -----
        ValueError, is the image_size and the patch_size are not consistent.
        '''
        
        
        if len(image_size) != (len(self.patch_size)):
            
            raise ValueError(f'image and pathc size must have the same dimenson: {len(image_size)} != {len(self.patch_size)}')
            
        if np.any(np.asarray(image_size) < np.asarray(self.patch_size)):
            raise ValueError(f'Patch must be contained inside the image: {image_size} < {self.patch_size}')