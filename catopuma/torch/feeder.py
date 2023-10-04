'''
'''
import numpy as np
import torch
from typing import List, Tuple, Optional, NoReturn

import catopuma
from catopuma.uploader import SimpleITKUploader
from catopuma.core.base import UploaderBase, PreProcessingBase, DataAgumentationBase


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = []



class FeederDataset(torch.utils.data.Dataset):

    def __init__(self, img_paths: List[str], target_paths: List[str],
                 uploader: UploaderBase = SimpleITKUploader(), 
                 preprocessing: Optional[PreProcessingBase] = None,
                 augmentation_strategy: Optional[DataAgumentationBase] = None):
        
        self.img_paths = np.asarray(img_paths)
        self.tar_paths = np.asarray(target_paths)
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

        return len(self.img_paths)

    def __getitem__(self, idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
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
        X, y = list(zip(*[self.uploader(img, tar) for img, tar in zip(self.img_paths[idxs], self.tar_paths[idxs])]))
        X = np.asarray(X)
        y = np.asarray(y)

        # now apply eventual data augmentation 
        if self.augmentation_strategy is not None:
            X, y = self.augmentation_strategy(X, y)

        # and the specified pre-processing
        if self.preprocessing is not None:
            X, y = self.preprocessing(X, y)

        return torch.Tensor(X.astype(np.float32)), torch.Tensor(y.astype(np.float32))



class ImageFeederOnTheFly(torch.utils.data.DataLoader):
    '''
    '''
    
    def __init__(self, img_paths: List[str], target_paths: List[str], batch_size: int = 8, shuffle: bool = True,
                uploader: UploaderBase = SimpleITKUploader(), 
                preprocessing: Optional[PreProcessingBase] = None,
                augmentation_strategy: Optional[DataAgumentationBase] = None):
        
        self.batch_size = batch_size

        # Here I am defining the dataset inside the init of the ImageFeederOnTheFly object. 
        # This is not so maintainable, but could simplify the whole repo usage,
        # However, consider to implement dependency inversion and injection pattern also in this case.
        # Howerr It could not be necessasy required
        self.dataset = FeederDataset(img_paths=img_paths, target_paths=target_paths,
                                     uploader=uploader, preprocessing=preprocessing, augmentation_strategy=augmentation_strategy)

        _ = self._checkConsistency()

        self.indexes = np.arange(0, len(img_paths), 1)
        self.shuffle = shuffle
        self.img_path = img_paths
        self.target_paths = target_paths
        self.uploader = uploader
        self.preprocessing = preprocessing
        self.augmentation_strategy = augmentation_strategy

        # and thes shuffle if specified
        _ = self.on_epoch_end()
    
    def _checkConsistency(self) -> NoReturn:
        '''
        Check the consistency of the provided init arguments:
            - len(img_paths) == len(tar_paths)
            - len(img_paths) >= batch_size

        If one of those requirements is not met, it will raise a value error.
        '''

        if len(self.dataset.img_paths) != len(self.dataset.tar_paths):
            raise ValueError(f'len of image paths must be the same of the targets: {len(self.dataset.img_paths)} != {len(self.dataset.tar_paths)}')
        if len(self.dataset.img_paths) < self.batch_size:
            raise ValueError(f'len of imagese must be at least equal to the batch size: {len(self.dataset.img_paths)} < {self.batch_size}')

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

        return len(self.dataset) // self.batch_size

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

        return self.dataset[idxs]
 
    def on_epoch_end(self):
        '''
        Function called at the end of each epoch.
        If self.shuffle is true, it will shuffle the images.
        '''

        if self.shuffle is True:
            np.random.shuffle(self.indexes)


    #
    # Define some calss methods to init the class in different ways, like from a directory,
    # from a dataframe or from an existing dataset
    #

    @classmethod
    def flow_from_directory(cls):
        pass

    @classmethod
    def from_dataset(cls):
        pass

    @classmethod
    def flow_from_dataframe(cls):
        pass

    @classmethod
    def flow_from_textfile(cls):
        pass