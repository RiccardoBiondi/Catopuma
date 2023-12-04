import catopuma
import numpy as np

from typing import Iterable, NoReturn, Tuple, List

from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK_BASE
from catopuma.core._padding_functions import get_padding_values_for_strides
from catopuma.core._padding_functions import pad_tensor

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']
__all__ = ['PatchPredict']


class PatchPredict:
    """
    Class to make a prediction in a patch-wise manner. 
    It will combine prediction makes by extracting a patch of a specified size and sliding it trough the image/ volume by specified strides.
    The results of this prediction will be summed and averaged according to the number of patches contributiong to the voxel prediction.
    The prediction cuold be carried out for a single image or on multple images loaded by a specified uploader.
    In both case it requires the image/ volume to be laoded, therefore no lazy loading is implemented (up to now)

    The whole code and approach is inspired by the one implemented by Gianluca Carlini(https://github.com/GianlucaCarlini/Segmentation3D/blob/main/utils/prediction_loops.py)
    
    Attributes
    ----------
    model: torch or tf.keras model (depending on your framework)
        trained model to use for the predicition.
        If pytorch model should have a forward method taking the input argument and returning the predicted activation map.
        if tf.keras or keras model, should implement a predict method with the same characteristics 
    patch_size:

    strides:

    padding:

    unpad:

    data_format:


    Methods
    ------- 

    predict_from_tensor:

    Usage
    -----

    For tf.keras or keras frameworks:

    >>> import os
    >>> os.
    >>> import catopuma
    >>> from catopuma.prediction_loops import PatchPredict
    >>> import SimpleITK as sitk
    >>>
    >>> # load keras model with weights
    >>> model = 
    >>>
    >>> # load the image to use for the prediction
    >>> image = sitk.ReadImage('/path/to/input/image.nii')
    >>> image = sitk.GetArrayFromImage(image)
    >>> image = (image - image.min()) / (image.max() - image.min())
    >>>
    >>> # convert the image into an array and normalize it. 
    >>>
    >>> # convert the image to a tensor
    >>>
    >>> # Finally make the prediction loop
    >>> with PatchPredict(model=model, strides=(8, 8), patch_size=(32, 32), padding="same") as pp:
    >>>     result = pp.predict_from_tensor(tensor)


    Or, on torch framework:

    >>> import os
    >>> os.
    >>> import torch
    >>> import catopuma
    >>> from catopuma.prediction_loops import PatchPredict
    >>> import SimpleITK as sitk
    >>>
    >>> # load the torch model
    >>> model = 
    >>> # and specify the evaluation mode on the desidered device (cuda or cpu)
    >>> mode.eval()
    >>> model.to('cuda')
    >>> # now load the image and create the tensor for the prediction
    >>> # ensure that the tensor is to the same device as the model
    >>> image = sitk.ReadImage('/path/to/input/image.nii')
    >>> image = sitk.GetArrayFromImage(image)
    >>> image = (image - image.min()) / (image.max() - image.min())
    >>> tensor = torch.from_numpy(image).to('cuda')
    >>>
    >>> # Finally make the prediction loop
    >>> with PatchPredict(model=model, strides=(8, 8), patch_size=(32, 32), padding="same") as pp:
    >>>     result = pp.predict_from_tensor(tensor)
    """

    def __init__(self,
                 model,
                 patch_size: Iterable[int],
                 strides: Iterable[int],
                 padding: str = "valid",
                 unpad: bool = True,
                 data_format: str = 'channels_last'):
        """
        """

        self._model = model
        self._patch_size = patch_size
        self._strides = strides
        self._unpad = unpad
        self._padding = padding
        self._data_format = data_format

        self._check_consistency()

    
    @property
    def model(self):
        """
        Model to use for the prediction.
        """
        return self._model


    @model.setter
    def model(self, value) -> NoReturn:
        """
        Setter that forbid user from changing the parameter once the
        object as been initialized
        """
        raise ValueError("PatchPredict model could not be changed")


    @property
    def strides(self) -> Iterable[int]:
        """
        Strides to slide the patch over the image/ volume
        """
        return self._strides


    @strides.setter
    def strides(self, value: Iterable[int]) -> NoReturn:
        """
        Setter that forbid user from changing the strides parameter once the
        object as been initialized
        """
        raise ValueError("PatchPredict strides could not be changed")


    @property
    def patch_size(self) -> Iterable[int]:
        """
        Size of the patches used to make the prediction
        """
        return self._patch_size
 

    @patch_size.setter
    def patch_size(self, value: Iterable[int]) -> NoReturn:
        """
        Setter that forbid user from changing the patch_size parameter once the
        object as been initialized
        """
        raise ValueError("PatchPredict patch_size could not be changed")

    @property
    def data_format(self):
        """
        """
        return self._data_format

    @data_format.setter
    def data_format(self, value: str) -> NoReturn:
        """
        Setter that forbid user from changing the data_format parameter once the
        object as been initialized
        """
        raise ValueError("PatchPredict data_format could not be changed")
    
    @property
    def padding(self) -> str:
        """
        """
        return self._padding

    @padding.setter
    def padding(self, value) -> NoReturn:
        """
        Setter that forbid user from changing the padding parameter once the
        object as been initialized
        """
        raise ValueError("PatchPredict padding could not be changed")

    @property
    def unpad(self):
        pass

    @unpad.setter
    def unpad(self, value):
        """
        Setter that forbid user from changing the unpad parameter once the
        object as been initialized
        """
        raise ValueError("unpad padding could not be changed")

    def _check_consistency(self) -> NoReturn:
        """
        Once the object attributes has been initialized, check if the the 
        initialization is coherent with the class requirements. 
        If not, a ValueError is returned.

        This function check that:

            - a valid data_format is passed
            - a valid padding method is passed
            - strides and patch_size have equal shape
            - strides and patch_size shape is 2 or 3 (images and volumes)

        Raise
        -----
        ValueError:
            raise a ValueError id any of the prevous conditions is not met.
        """


        if len(self.patch_size) != len(self.strides):
            raise ValueError(f"Patch size and strides must have the same len: {self.patch_size} != {self.strides}")
        
        if len(self.patch_size) not in [2, 3]: 
            raise ValueError(f"patch_size shape {len(self.patch_size)} not supported. Use 2 or 3 instead")


    def __enter__(self):
        """
        Enter method to use the objec as context manager. 
        It will return self. 
        """

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit method to make the object usable as context manager.
        """
        pass

    def __call__(self):
        pass

    def _drop_channel_index(self, tensor_shape: Tuple[int]) -> Tuple[int]:
        '''
        Drop the channel index according to the data fromat

        Parameter
        ---------
        tensor_shape: Tuple[int]
            shape of the input tensor tensor.
            The shape must be (n_channels, w, h, d) or (n_channels, w, h) if data_format is channels_first of 
            (w, h, d, n_channels) or (w, h, n_channels) if data_format is channels_last.

        Return
        ------
        dropped_shape: Tuple[int]
            tensor shape without the channel dimension.
        '''
        
        if self.data_format == 'channels_last':
            
            return tensor_shape[:-1]

        return tensor_shape[1:]

    def _get_patch_coords(self, padded_tensor_shape: Tuple[int]) -> Tuple[List[int]]:
        '''
        Compute the top and bottom coordinates on each direction for all the patches.

        Parameter
        ---------
        padded_tensor_shape: Tuple[int]
            tuple with the shape of the padded tensor without the channel dimension.

        Return
        ------
        corners Tuple[List[int]]
            tuple with the list of top and bottom corners.
            In this case also channel indexes are incorporated

        '''

        # compute the numner of patches in each direction
        n_patches = [int((d - p) / s) + 1 for d, p, s in zip(padded_tensor_shape, self.patch_size, self.strides)]

        # for each direction create the range of values
        ranges = [np.arange(0, n_patch, 1) for n_patch in n_patches]


        # now create the mesh grid to use to compute the patches coordinates
        indexes = np.meshgrid(*ranges, indexing="ij")

        # find the initial corners
        coords_top_corner = [i.ravel() * s for i, s in zip(indexes, self.strides)]
        coords_bottom_corner = [i.ravel() + p for i, p in zip(coords_top_corner, self.patch_size)]

        
        # now prepare and add the list of indexes for the channel.
        channel_indexes = len(coords_top_corner[0]) * [None]

        if self.data_format == 'channels_first':
            coords_bottom_corner.inser(0, channel_indexes)
            coords_top_corner.insert(0, channel_indexes)

        else:
            coords_bottom_corner.append(channel_indexes)
            coords_top_corner.append(channel_indexes)
        
        coords_bottom_corner = np.stack(coords_bottom_corner)
        coords_top_corner = np.stack(coords_top_corner)

        coords_bottom_corner = coords_bottom_corner.transpose(1, 0)
        coords_top_corner = coords_top_corner.transpose(1, 0)

        return coords_top_corner, coords_bottom_corner


    def _prediction_step(self, top, bottom, padded_tensor) -> None:
        '''
        Prediction step for a single patch. 
        It will make the prediction for the selected patch and update the mask.
        To make the prediction it will call
            - model.predict(patch), if the framework si tf.keras/keras
            - model(patch), if the framework is torch  
        '''

        slices = tuple([slice(t, b) for t, b in zip(top, bottom)])
        input_ = padded_tensor[slices]



        # add a fucntion to make the prediction
        if _FRAMEWORK_NAME == 'torch':
            res = self.model(input_)
        else:
            input_ = _FRAMEWORK_BASE.expand_dims(input_, 0)
            res = self.model(input_)
            res = _FRAMEWORK_BASE.reshape(res, res.shape[1:])

        
        # update also the mask and the prediction
        self.mask[slices] = self.mask[slices] + 1
        self.pred[slices] = self.pred[slices] + res

    def _unpad_tensor(self, tensor, pad_values) -> None:
        '''
        Unpad the image to make it to the original shape.

        Parameters
        ----------
        tensor: tensor
            tensor to unpad

        pad_values: Iterable[Iterable[int]]
            pad values used to pad the tensor in the same modality
        
        Return
        ------
        unpadded: tensort
            the unpadded input tensor
        '''
        print(pad_values)
        slices = tuple([slice(up, -down) for up, down in pad_values])

        return tensor[slices]

    def predict_from_tensor(self, X):
        """
        Make the patch-wise prediction on the provided image/ volume.
        The prediction in made by patching it in a sliding window fashion.

        Parameter
        ---------
        X: tensor
            input tensor to predict. 
            The shape must be (n_channels, w, h, d) or (n_channels, w, h) if data_format is channels_first of 
            (w, h, d, n_channels) or (w, h, n_channels) if data_format is channels_last.
            The prediction is made on a single image, no batch dimension is allowed
        
        Return
        ------
        pred: tensor
            output prediction tensor with the same shape of the input one.
        """

        # get the image shape and drop the channel one
        array_shape = np.asarray(X.shape)
        tensor_shape = self._drop_channel_index(array_shape)
        # get the eventual padding dimensions

        pad_values = get_padding_values_for_strides(array_shape=tensor_shape, patch_size=self.patch_size,
                                                    strides=self.strides, padding=self._padding)

        # then pad the array
        padded_tensor = pad_tensor(X, pad_values, self._padding) 
        padded_tensor_shape = self._drop_channel_index(np.asarray(padded_tensor.shape))
        # make the zero valued array for the prediction
        # and the mask to normalize the array
        self.pred = np.zeros(shape=padded_tensor.shape)
        self.mask = np.zeros(shape=padded_tensor.shape)

        # now compute the number of patches and the patch corners
        top_corners, bottom_corners = self._get_patch_coords(padded_tensor_shape)

        for top, bottom in zip(top_corners, bottom_corners):

            self._prediction_step(top, bottom, padded_tensor)
        # now normalize the prediction to obtain a valid activation map

        
        self.pred = self.pred / self.mask
        # and, if required, unpad the image
        if self._unpad & (self.padding == 'same'):
            self.pred = self._unpad_tensor(self.pred, pad_values)

        # finally return the result

        return self.pred