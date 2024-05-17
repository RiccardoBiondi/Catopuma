import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from  hypothesis import HealthCheck as HC


import numpy as np
from typing import List
#import tensorflow as tf

import catopuma
from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK as F
from catopuma.core.__framework import _FRAMEWORK_BASE as B
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K
from catopuma.core.__framework import _DATA_FORMAT
# import the class to test
from catopuma.prediction_loops import PatchPredict

# create a dummy model to test the class.
# the model is created according to the used workflow

def get_dummy_model(shape=None):
    '''
    Return a simple model according to the selected workflow 
    to test the PatchPredict class.
    The dummy model contains only a ReLU activation layer to 
    allows the model to get as output an image euqual to the 
    input one.
    In this way should be easier to test the functionalities.
    '''

    if _FRAMEWORK_NAME in ['keras', 'tf.keras']:

        inputs = F.Input(shape=shape)
        outputs =  F.layers.Identity()(inputs)
        model = F.Model(inputs=inputs, outputs=outputs)
        return model
    

    class DummyModel(F.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()

            self.identity = F.nn.Identity()

        def forward(self, x):
            x = self.identity(x)            
            return x

    dummymodel = DummyModel()

    return dummymodel
#
# Start testing the exceptions
#

@given(st.lists(st.integers(32, 64), min_size=1, max_size=5), st.lists(st.integers(8, 16), min_size=6, max_size=10))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_patch_predict_raise_value_error_if_patch_strides_different_shapes(patch_size, strides):
    '''
    Test that a value error is raised if PatchPredict is initiazlied with strides and patch_size of different len

    Given
    -----
        - dummy model
        - random patch_size
        - random strides with len != patch_size

    Then
    ----
        - init the PatchPredict class

    Assert
    ------
        - value error is raised
    '''

    model = get_dummy_model((3,))

    with pytest.raises(ValueError):

        obj = PatchPredict(model=model, patch_size=patch_size, strides=strides)


@given(st.lists(st.integers(32, 64), min_size=4, max_size=10))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_patch_predict_raise_value_error_if_patch_wrong_len(patch_size):
    '''
    Test that a value error is raised if PatchPredict is initiazlied with len different of 2 or 3.

    Given
    -----
        - dummy model
        - random patch_size with len > 3
        - strides setted equal to patch_size to avoide raising other exceptions
    Then
    ----
        - init the PatchPredict class

    Assert
    ------
        - value error is raised
    '''

    model = get_dummy_model((3,))

    with pytest.raises(ValueError):

        obj = PatchPredict(model=model, patch_size=patch_size, strides=patch_size)


#
# Now Check the attributes
#


#
# An finally the methods
#


def test_drop_channel_index_channels_last_2d():
    """
    Test that the correct cannel dimension is dropped when given the 
    tensor shape in (h, w, n_channels), data_format channels_last

    Given
        - dummy model 2d strides, and patch_size to correctly initialize PatchPredict
        - tensor_shape == (128, 128, 3)

    Then
        - init the PatchPredict object
        - call the _drop_channel_index method
    Assert
        - _drop_channel_index return (128, 128)
    """
    dummy_model = get_dummy_model((3,))
    obj = PatchPredict(model=dummy_model, strides=(8, 8), patch_size=((64, 64)), data_format='channels_last')
    
    res = obj._drop_channel_index((128, 128, 3))

    assert res == (128, 128)

def test_drop_channel_index_channels_last_3d():
    """
    Test that the correct cannel dimension is dropped when given the 
    tensor shape in (h, w, d, n_channels), data_format channels_last

    Given
        - dummy model 2d strides, and patch_size to correctly initialize PatchPredict
        - tensor_shape == (128, 128, 128, 3)

    Then
        - init the PatchPredict object
        - call the _drop_channel_index method
    Assert
        - _drop_channel_index return (128, 128, 128)
    """
    dummy_model = get_dummy_model((3,))
    obj = PatchPredict(model=dummy_model, strides=(8, 8, 8), patch_size=((64, 64, 64)), data_format='channels_last')
    
    res = obj._drop_channel_index((128, 128, 128, 3))

    assert res == (128, 128, 128)


def test_drop_channel_index_channels_first_2d():
    """
    Test that the correct cannel dimension is dropped when given the 
    tensor shape in (n_channels, h, w) , data_format channels_first

    Given
        - dummy model 2d strides, and patch_size to correctly initialize PatchPredict
        - data_format channels_first
        - tensor_shape == (3, 128, 128)

    Then
        - init the PatchPredict object
        - call the _drop_channel_index method
    Assert
        - _drop_channel_index return (128, 128)
    """
    dummy_model = get_dummy_model((3,))
    obj = PatchPredict(model=dummy_model, strides=(8, 8), patch_size=((64, 64)), data_format='channels_first')
    
    res = obj._drop_channel_index((3, 128, 128))

    assert res == (128, 128)



def test_drop_channel_index_channels_first_3d():
    """
    Test that the correct cannel dimension is dropped when given the 
    tensor shape in (n_channels, h, w, d), data_format channels_first

    Given
        - dummy model 2d strides, and patch_size to correctly initialize PatchPredict
        - data_format == channels_first
        - tensor_shape == (3, 128, 128, 128)

    Then
        - init the PatchPredict object
        - call the _drop_channel_index method
    Assert
        - _drop_channel_index return (128, 128, 128)
    """
    dummy_model = get_dummy_model((3,))
    obj = PatchPredict(model=dummy_model, strides=(8, 8, 8), patch_size=((64, 64, 64)), data_format='channels_first')
    
    res = obj._drop_channel_index((3, 128, 128, 128))

    assert res == (128, 128, 128)


@given(st.integers(2, 3))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_get_patch_coords_bottom_top_coords_equal_shape(shape):
    '''
    Given:
        - shape of the image (2D or 3D)
    Than:
        - Initialize tensor shape, strides and padding
        - Initialize the PatchPredict object
        - compute the top and bottom patch coords

    Assert:
        - top and bootom coords have the same shape
        - assert the first second dimension is equal to th shape of the image + 1 (the channel dimension)
    '''
    # create the shape of strides, patch size and input tensor
    # shape are diffrent in each dimension
    strides = tuple(map(lambda x: 8 * 2**x, range(shape)))
    patch = tuple(map(lambda x: 32 * 2**x, range(shape)))
    tensor = tuple(map(lambda x:  64 * 2**x, range(shape)))
    
    # get the dummy model to initialie the object
    model = get_dummy_model((3,))

    # and finally intialize it
    obj = PatchPredict(model=model, patch_size=patch, strides=strides)

    # get the patch coords by the methods
    top_coord, bottom_coords = obj._get_patch_coords(tensor)
    
    # and finally make the assertion

    assert top_coord.shape == bottom_coords.shape
    assert top_coord.shape[1] == bottom_coords.shape[1] == shape + 2 # -2 to consider also the channel and tha batch dimension


def test_unpad_tensor_return_correct_shaped_tensor_channels_first():
    '''
    Test that the unpad tensor method return a tensor with the expected shape in the 
    channels_first data format case.
    Ideally the function should be agnostic to the data format, sicne the index values are
    given from the outside, however I will check for both the data format cases.

    Given
    -----
        - 
    Then
    ----
        - 
    Assert
    ------
        - 

    '''
    pass


def test_unpad_tensor_return_correct_shaped_tensor_channels_last():
    '''
    Test that the unpad tensor method return a tensor with the expected shape in the 
    channels_last data format case.
    Ideally the function should be agnostic to the data format, sicne the index values are
    given from the outside, however I will check for both the data format cases.

    Given
    -----
        - 
    Then
    ----
        - 
    Assert
    ------
        - 
    '''
    pass


@given(st.integers(2, 3), st.integers(16, 32), st.integers(64, 128))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpadded_prediction_equal_input_for_identity_model(image_dim, strides, patch_size):
    '''
    Test that the predicted image is equal to the input one when the model
    return the identity during the prediction (i.e. the dummy model implemented when the input image have only posivie voxel values).

    During the prediction the unpad argument is set to True and the padding to 'same' to make the predictor returning an image with the 
    same dimension of the input one. 

    Given
    -----
        - 2D/3D image shape
        - patch size
        - strides
    Then
    ----
        - construct an input image/volume with the given shape and positive voxel values
        - instantiate the DummyModel
        - apply the patch prediction with the specified patch_size and strides and unpad=True, padding='same'
        - compute the MAE between the predicted and the input image
    Assert
    ------
        - the input and output image are equal
    '''

    sizes = image_dim * [256]
    sample = 255. * np.random.rand(*sizes) # add one to avoid problem with the 0., since the activation is a relus

    # now convert to inger and then back to float. 
    # this will reduce the issues arising from the floating point 
    # precision.
    
    sample = sample.astype(np.uint8)
    sample = sample.astype(np.float32)
    
    # now add the channel dimension according to 
    # the specified data format
    if _DATA_FORMAT == 'channels_first':
        sample = sample[np.newaxis]
    else:
        sample = sample[..., np.newaxis]

    
    # now convert the array to a tensor 
    if _FRAMEWORK_NAME == 'torch':
        sample = F.Tensor(sample)
    else:
        sample = B.convert_to_tensor(sample)

    # now make the prediction
    strides_ = image_dim * [strides]
    patch_size_ = image_dim * [patch_size]
    model = get_dummy_model([*patch_size_, 1])
    # and meke the prediction
    with PatchPredict(model=model, strides=strides_, patch_size=patch_size_, padding='same', unpad=True) as pp:
        pred = pp.predict_from_tensor(sample)   
    # remove the batch dimension from the prediction
    assert np.all(np.isclose(pred[0], sample))