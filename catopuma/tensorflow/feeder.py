'''
'''
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, NoReturn

import catopuma
from catopuma.uploader import SimpleITKUploader
from catopuma.core.base import UploaderBase, PreProcessingBase, DataAgumentationBase


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = ['ImageFeederOnTheFly']


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
    paths: List[str]
        list of strings. Each list must contain the paths to the image to load.
        At least two lists ,ust be provided, one for imagesand one for targets.
        The last list provided will be considered as the taget paths list, all the 
        others as the input image paths.
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

    Example
    -------
    >>>
    >>>
    '''

    def __init__(self,
                *paths,
                batch_size: int = 8, shuffle: bool = True,
                uploader: UploaderBase = SimpleITKUploader(), 
                preprocessing: Optional[PreProcessingBase] = None,
                augmentation_strategy: Optional[DataAgumentationBase] = None) -> None:

        # start the initialization of the feeder
        self.paths = paths
        self.batch_size = batch_size

        # check the input data consistency
        _ = self._checkConsistency()

        # if pass the check fase, finish the initalization 
        self.uploader = uploader
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.augmentation_strategy = augmentation_strategy

        # define the indexes for the path access
        self.indexes = np.arange(0, len(paths[0]), 1)

        # and thes shuffle if specified
        self.on_epoch_end()

    def _checkConsistency(self) -> NoReturn:
        '''
        Check the consistency of the provided init arguments:
            - len paths >= 2 (at least a path to a single input image and to a target image
            should be provided)
            - all the paths lists must have the same lenght
            - len(paths[0]) >= batch_size

        If one of those requirements is not met, it will raise a value error.
        '''

        if len(self.paths) < 2:
            raise ValueError(f'At least two list of paths must be provided. Provided {len(self.paths)}')
        
        for path in self.paths[1:]:
            if len(self.paths[0]) != len(path):
                raise ValueError(f'All the provided image paths must be the same: {len(self.paths[0])} != {len(path[0])}')

        if len(self.paths[0]) < self.batch_size:
            raise ValueError(f'len of imagese must be at least equal to the batch size: {len(self.paths[0])} < {self.batch_size}')

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
        X, y = list(zip(*[self.uploader(*elem) for elem in zip(*[np.asarray(p)[idxs] for p in self.paths])]))
        X = np.asarray(X)
        y = np.asarray(y)

        # now apply eventual data augmentation 
        if self.augmentation_strategy is not None:
            X, y = self.augmentation_strategy(X, y)

        # and the specified pre-processing
        if self.preprocessing is not None:
            X, y = self.preprocessing(X, y)

        return tf.convert_to_tensor(X, dtype='float'), tf.convert_to_tensor(y, dtype='float')

    def on_epoch_end(self):
        '''
        Function called at the end of each epoch.
        If self.shuffle is true, it will shuffle the images.
        '''

        if self.shuffle:
            np.random.shuffle(self.indexes)


    @classmethod
    def flow_from_directory(cls, directory: str, batch_size: int = 8, shuffle: bool = True,
                            uploader: UploaderBase = SimpleITKUploader(), preprocessing: PreProcessingBase = None,
                            data_augmentation: DataAgumentationBase = None):
        '''
        Class method to create a feeder from a directory.

        Parameters
        ----------
        directory: str
            the path to the directory containing the images and the targets.
            The images and the targets must be in the same directory, and the targets must have the same name of the
            corresponding image, with the only difference being the extension.
        '''
        pass