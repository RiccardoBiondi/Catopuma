import os
import catopuma
import numpy as np
import albumentations as A
import volumentations as V
from catopuma.core.base import DataAgumentationBase


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


class AlbumentationDataAugmentation2D(DataAgumentationBase):
    '''
    Perform the data augmentation on a 2D image using the albumentation library.

    Attributes
    ----------
    transform: list
        A list of albumentation transforms to apply to compose and to apply on the images

    Example
    -------
    >>> import catopuma
    >>> import cv2
    >>> import albumentations as A
    >>> from catopuma.augmentation import AlbumentationDataAugmentation2D
    >>> from catopuma.feeder import ImageFeederOnTheFly
    >>>
    >>> # get the path to the images and targets
    >>> img_paths = ['img1.nii', img2.nii', 'img3.nii']
    >>> tar_paths = ['tar1.nii', tar2.nii', 'tar3.nii']
    >>>
    >>> # define the augmentation strategies list
    >>> augmentation_transforms = [A.Rotate(p=.8, limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, value=(-4, -4),  mask_value=0)]
    >>>
    >>> # And instantiate the data augmentation class
    >>> augmentation = AlbumentationDataAugmentation2D(augmentation_transforms)
    >>>
    >>> # Finally init the feeder 
    >>> feeder = ImageFeederOnTheFly(img_paths, tar_paths, augmentation_strategy=augmentation)
    '''
    def __init__(self, transform : list):

        self.transform = A.Compose(transform)


    def __call__(self, image, mask):

        # first of all, check if the batch dimension is specified
        # if not, then apply the augmentation directly to the image
        if len(image.shape) == 3:

            sample = self.transform(image=image, mask=mask)

            return sample['image'], sample['mask']

        # otherwise, perform the augmentation on each image in the batch
        else:
        
            X = []
            y = []

            for im, ms in zip(image, mask):

                sample = self.transform(image=im, mask=ms)
                X.append(sample['image'])
                y.append(sample['mask'])

            return np.asarray(X), np.asarray(y)


class VolumentationDataAugmentation3D(DataAgumentationBase):
    '''
    Perform the data augmentation on a 3D volumes using the volumentations-3D library.

    Attributes
    ----------
    transform: list
        A list of volumebntations transforms to apply to compose and to apply on the images

    Example
    -------
    >>> import catopuma
    >>> import cv2
    >>> import volumebntations as V
    >>> from catopuma.augmentation import VolumentationDataAugmentation3D
    >>> from catopuma.feeder import ImageFeederOnTheFly
    >>>
    >>> # get the path to the images and targets
    >>> vol_paths = ['vol1.nii', vol2.nii', 'vol3.nii']
    >>> tar_paths = ['tar1.nii', tar2.nii', 'tar3.nii']
    >>>
    >>> # define the augmentation strategies list
    >>> augmentation_transforms = [V.Rotate((-15, 15), (0, 0), (0, 0), p=0.5)]
    >>>
    >>> # And instantiate the data augmentation class
    >>> augmentation = VolumentationDataAugmentation2D(augmentation_transforms)
    >>>
    >>> # Finally init the feeder 
    >>> feeder = ImageFeederOnTheFly(vol_paths, tar_paths, augmentation_strategy=augmentation)
    '''
    def __init__(self, transform : list):

        self.transform = V.Compose(transform)


    def __call__(self, image, mask):

        # first of all, check if the batch dimension is specified
        # if not, then apply the augmentation directly to the volume
        if len(image.shape) == 4:

            sample = self.transform(image=image, mask=mask)

            return sample['image'], sample['mask']

        # otherwise, perform the augmentation on each image in the batch
        else:
        
            X = []
            y = []

            for im, ms in zip(image, mask):

                sample = self.transform(image=im, mask=ms)
                X.append(sample['image'])
                y.append(sample['mask'])

            return np.asarray(X), np.asarray(y)
