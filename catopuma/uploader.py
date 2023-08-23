'''
Module containing the implementation of the vaious Uploader to read and organize the input image
and tergetes.
'''

from typing import Tuple, NoReturn, Dict
import numpy as np
import SimpleITK as sitk

from catopuma.core.base import UploaderBase


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

class SimpleITKUploader(UploaderBase):
    '''
    Class to upload medical image formats, based on SimpleITK image reader;
    therefore the allowed data format are the ones specified in https://simpleitk.readthedocs.io/en/master/IO.html.
    This reader allows to read both images and 3D volumes.

    Parameters
    ----------
    data_format: str (default 'channels_last')
        Specify the shape of the returned image or volume.
        Can be either 'channels_first', i.e. (channels, ...), or channels_last, i.e. (..., channels). 
    '''

    EXPANSION_AXIS: Dict[str, int] = {
        'channels_last': -1,
        'channels_first': 0}

    def __init__(self, data_format: str = 'channels_last') -> None:

        super().__init__()
        if data_format in self.EXPANSION_AXIS:
            self.data_format = data_format
        else:
            raise ValueError(f'{data_format} is not allowed. Allowed data formats:  {self.EXPANSION_AXIS.keys()}')

    def __call__(self, *path: Tuple[str]) -> Tuple[np.ndarray]:
        '''
        '''
        img = sitk.ReadImage(path[0]) # input path must be first
        tar = sitk.ReadImage(path[1]) # target path must be second

        # convert the images to array
        img = sitk.GetArrayFromImage(img)
        tar = sitk.GetArrayFromImage(tar)

        img = np.expand_dims(img, axis=self.EXPANSION_AXIS[self.data_format])
        tar = np.expand_dims(tar, axis=self.EXPANSION_AXIS[self.data_format])

        return img, tar


class LazyPatchBaseUploader(UploaderBase):
    '''
    Loader to peform a patch base lazy loading of medical images. 
    The loader is based on the SimpleITK reader, therefore the allowed data format are specified 
    in https://simpleitk.readthedocs.io/en/master/IO.html

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

    EXPANSION_AXIS: Dict[str, int] = {
        'channels_last': -1,
        'channels_first': 0}

    def __init__(self, patch_size: Tuple[int], threshold: float = -1., data_format: str = 'channels_last') -> None:

        super().__init__()
        if data_format in self.EXPANSION_AXIS.keys():
            self.data_format = data_format
        else:
            raise ValueError(f'{data_format} is not allowed. Allowed data formats:  {self.EXPANSION_AXIS.keys()}')
            
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
            y = sitk.GetArrayFromImage(reader.Execute())
            
            if np.sum(y > 0.0) > self.threshold * np.prod(np.asarray(self.patch_size)):
                condition = False
            
        _ = reader.SetFileName(path[0])
        X = sitk.GetArrayFromImage(reader.Execute())

        X = np.expand_dims(X, axis=self.EXPANSION_AXIS[self.data_format])
        y = np.expand_dims(y, axis=self.EXPANSION_AXIS[self.data_format])

        return X, y


    def _samplePatchOrigin(self, image_size: Tuple[int]) -> Tuple[int]:
        '''
        Sample the patch origin according to a uniform distribution ensuring 
        that all the patch is inside the the image
        
        Parameters
        ----------
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