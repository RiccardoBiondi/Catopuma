'''
'''

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, NoReturn

from catopuma.uploader import SimpleITKUploader
from catopuma.core.abstraction import UploaderBase, PreProcessingBase, DataAgumentationBase

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

class ImageFeederOnTheFly(tf.keras.utils.Sequence):
    '''
    Class to feed the images to the deep Learning model. 
    This class is implemented to load each batch at run time, saving memory when the whole 
    sample is too long and too heavy. 
    It also has the possibility to augment and preprocess the input data and targets on the fly, 
    before presenting the batch to the Model.
    It also allows to specify different readers for different kind of images, and for different 
    reading modalities.

    Parameters
    ----------
    img_path: list[str]
        list of paths to the input images.
    tar_path: list[str]
        list of paths to the target images. Must be of the same length of img_path.
        Note that the order is important: the target_path at position idx must be the one corresponding 
        to the img_path at position idx.
    batch_size: int (default 8)
        the size of each batch: must be lower or equal to the length of the image path.
    shuffle: bool (default True)
        specify if shuffle or not the path order at the end of each epoch
    uploader: UploaderBase (default SimpleITKUploader)
        the uploader used to load image and targets. Must return a X, y pair where X, y and the input and the 
        target images. Custom uploader must inherit from UploaderBase
    preprocessing: PreProceessingBase (default None)
        if specified, the class specifying the preprocessing routine. Must return a pair X, y and inherit from
        PreProcessingBase
    augmentation_strategy: DataAugmentationBase (default None)
        class implemeting the strategies to augment the data on the fly.
        Must return a pair X, y and inherit from DataAgumentationBase
    '''

    def __init__(self, img_paths: List[str], target_paths: List[str],
                 batch_size: int = 8, shuffle: bool = True,
                 uploader: UploaderBase = SimpleITKUploader(), 
                 preprocessing: Optional[PreProcessingBase] = None,
                 augmentation_strategy: Optional[DataAgumentationBase] = None) -> None:
    
        # start the initialization of the feeder
        self.img_paths = np.asarray(img_paths)
        self.tar_paths = np.asarray(target_paths)
        self.batch_size = batch_size
    
        # check the input data consistency
        _ = self._checkConsistency()
        
        # if pass the check fase, finish the initalization 
        self.uploader = uploader
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.augmentation_strategy = augmentation_strategy

        # define the indexes for the path access
        self.indexes = np.arange(0, len(img_paths), 1)
        # and thes shuffle if specified
        self.on_epoch_end()

    def _checkConsistency(self) -> NoReturn:
        '''
        Check the consistency of the provided init arguments:
            - len(img_paths) == len(tar_paths)
            - len(img_paths) >= batch_size

        If one of those requirements is not met, it will raise a value error.
        '''
        if len(self.img_paths) != len(self.tar_paths):
            raise ValueError(f'len of image paths must be the same of the targets: {len(self.img_paths)} != {len(self.tar_paths)}')
        
        if len(self.img_paths) < self.batch_size:
            raise ValueError(f'len of imagese must be at least equal to the batch size: {len(self.img_paths)} < {self.batch_size}')

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
        return len(self.indexes) // self.batch_size

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
        idxs = self.indexes[self.batch_size * idx: self.batch_size * (idx + 1)]

        # load the image and the labels
        X, y = np.asarray(list(zip(*[self.uploader(img, tar) for img, tar in zip(self.img_paths[idxs], self.tar_paths[idxs])])))

        # now apply eventual data augmentation 
        if self.augmentation_strategy is not None:
            X, y = self.augmentation_strategy(X, y)

        # and the specified pre-processing
        if self.preprocessing is not None:
            X, y = self.preprocessing(X, y)

        return X, y

    def on_epoch_end(self):
        '''
        Function called at the end of each epoch.
        If self.shuffle is true, it will shuffle the images.
        '''
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

