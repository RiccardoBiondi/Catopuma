'''
'''
import numpy as np
import torch
from typing import List, Tuple, Optional, NoReturn

import catopuma
from catopuma.uploader import SimpleITKUploader
from catopuma.core.base import UploaderBase, PreProcessingBase, DataAgumentationBase

from catopuma.core.__framework import _FRAMEWORK_BASE
from catopuma.core.__framework import _DATA_FORMAT


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = []



class ImageFeederOnTheFly(torch.utils.data.Dataset):
    '''
    Class implementing the Dataset to load the images to the deep Learning model. 
    This class is implemented to load each image at run time, saving memory when the whole 
    sample is too long and too heavy.
    It also has the possibility to augment and preprocess the input data and targets on the fly, 
    before presenting the image to DataLoader and the Model.
    It also allows to specify different readers for different kind of images and for different 
    reading modalities.
    Parameters
    ----------
    paths: List[str]
        list of strings. Each list must contain the paths to the image to load.
        At least two lists ,ust be provided, one for imagesand one for targets.
        The last list provided will be considered as the taget paths list, all the 
        others as the input image paths.
    uploader: UploaderBase (default SimpleITKUploader)
        the uploader used to load image and targets. Must return a X, y pair where X, y and the input and the 
        target images. Custom uploader must inherit from UploaderBase
    preprocessing: PreProceessingBase (default None)
        if specified, the class specifying the preprocessing routine. Must return a pair X, y and inherit from
        PreProcessingBase
    augmentation_strategy: DataAugmentationBase (default None)
        class implemeting the strategies to augment the data on the fly.
        Must return a pair X, y and inherit from DataAgumentationBase

    Example
    -------
    >>> import os
    >>> os.environ['CATOPUMA_FRAMEWORK'] = 'torch'
    >>> import catopuma
    >>> from catopuma.uploader import SimpleITKUploader
    >>> from catopuma.preprocessing import PreProcessing
    >>> import catopuma.feeder as feeder
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # define the paths to the input image and the target images
    >>> img_paths = ['img1.nii', 'img2.nii', 'img3.nii']
    >>> tar_paths = ['tar1.nii', 'tar2.nii', 'tar3.nii']
    >>> 
    >>> # define the pre-processing to apply to the image (rescale the image into [0., 1.])
    >>> preprocessing = PreProcessing(standardizer='minmax', target_standardizer='identity')
    >>> feed = feeder(img_path, tar_path, preprocessing=preprocessing)
    >>> 
    >>> # show the loaded images
    >>> for i in range(0, 3):
    >>>     X, y = feed[i]
    >>>     fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    >>>     _ = ax0.imshow(X, cmap='gray')
    >>>     _ = ax1.imshow(y, cmap='gray')
    >>>
    >>> # now define the DataLoader to feed the batch to the model
    >>> train_dataloader = DataLoader(feed, batch_size=2, shuffle=True)
    '''

    def __init__(self, *paths,
                uploader: UploaderBase = SimpleITKUploader(), 
                preprocessing: Optional[PreProcessingBase] = None,
                augmentation_strategy: Optional[DataAgumentationBase] = None):
        
        self.paths = paths

        self._checkConsistency()

        self.uploader = uploader
        self.preprocessing = preprocessing
        self.augmentation_strategy = augmentation_strategy

    def __len__(self) -> int:
        '''
        Return the number batches.
        This function returns only the numbers of complete batches.
        Batches with a number of examples lower than batch_size are not considered

        Return
        ------
        len: int
            number of full batches
        '''

        return len(self.paths[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Return the batch corresponding of idx.

        Parameters
        ----------
        idx: int
            The index corresponding to the desired batch.
            Must be lower than the number of full batches

        Return
        ------
        X, y: Tuple[np.ndarray, np.ndarray]
            X input images, y corresponding target of the batch
        '''

        # get the list of index for the batch

        # load the image and the labels
        X, y = self.uploader(*[p[idx] for p in self.paths])
        # add batch dimension to keep it consistent with the pre-processing step
        # it will be removed at the end of the process
        X = np.asarray(X)[np.newaxis]
        y = np.asarray(y)[np.newaxis]

        # now apply eventual data augmentation 
        if self.augmentation_strategy is not None:
            X, y = self.augmentation_strategy(X, y)

        # and the specified pre-processing
        if self.preprocessing is not None:
            X, y = self.preprocessing(X, y)

        X = np.squeeze(X, 0)
        y = np.squeeze(y, 0)

        return torch.Tensor(X.astype(np.float32)), torch.Tensor(y.astype(np.float32))

    def _checkConsistency(self) -> NoReturn:
        '''
        Check the consistency of the provided init arguments:
            - len paths >= 2 (at least a path to a single input image and to a target image
            should be provided)
            - all the paths lists must have the same lenght

        If one of those requirements is not met, it will raise a value error.
        '''

        if len(self.paths) < 2:
            raise ValueError(f'At least two list of paths must be provided. Provided {len(self.paths)}')
        
        for path in self.paths[1:]:
            if len(self.paths[0]) != len(path):
                raise ValueError(f'All the provided image paths must be the same: {len(self.paths[0])} != {len(path[0])}')