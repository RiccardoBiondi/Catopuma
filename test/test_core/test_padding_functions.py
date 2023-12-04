import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from  hypothesis import HealthCheck as HC

import numpy as np
from typing import List

import catopuma

from catopuma.core.__framework import _DATA_FORMAT
from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK_BASE as B
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K

from catopuma.core._padding_functions import _CHANNEL_AXIS
# import the functions to test
from catopuma.core._padding_functions import _get_same_padding_values_for_strides
from catopuma.core._padding_functions import _get_valid_padding_values_for_strides
from catopuma.core._padding_functions import get_padding_values_for_strides
from catopuma.core._padding_functions import _pad_tensor_same
from catopuma.core._padding_functions import _pad_tensor_valid
from catopuma.core._padding_functions import pad_tensor

# a simple dictionary to store the allowed padding modalities for testing purposes
ALLOWED_PADDING_MODALITIES: List[str] = ['valid', 'same']


# text strategies to generate invalid padding modalities
legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)


#
# Start the testing the raising of the exceptions
#

@given(text_strategy)
def test_get_padding_values_for_strides_raise_value_error_not_allowed_padding_mode(padding_mode):
    """
    Test that a value error is raised when an invalid padding modality is provided

    Given
    -----
        - random text different from any allowed padding modality
    Then
    ----
        - call the get_padding_values_for_strides

    Assert
    ------
        - value error is raised
    """
    assume(padding_mode not in ALLOWED_PADDING_MODALITIES)
    with pytest.raises(ValueError):
        val = get_padding_values_for_strides((0., 0., 0), (0., 0., 0.), (0., 0., 0.), padding=padding_mode)


@given(st.lists(st.integers(1, 10), min_size=1, max_size=5),
        st.lists(st.integers(1, 10), min_size=1, max_size=5), 
        st.lists(st.integers(1, 10), min_size=1, max_size=5))
def test_get_padding_values_for_strides_raise_value_error_different_len(array_shape, strides, patch_size):
    """
    Test that the get_padding_values_for_strides_raise_value_error_different_shapes correctly raise value
    error when the lenght of the provided array_shape and strieds does not match

    Given
    -----
        - list of random int for the array shape
        - list of random int for the strides with different lenght of the array_shape one
        - ist of random int for the strides with different lenght of the array_shape one
    Then
    ----
        - call get_padding_values_for_strides

    Assert
    ------
        - value error is raised
    """

    assume((len(array_shape) != len(strides)) | (len(patch_size) != len(array_shape)))

    with pytest.raises(ValueError):
        val = get_padding_values_for_strides(array_shape=array_shape, strides=strides, patch_size=patch_size)


@given(text_strategy)
def test_pad_tensor_raise_value_error(padding_mode):
    '''
    Test that a value error is raised when an invalid padding modality is provided

    Given
    -----
        - random text different from any allowed padding modality
    Then
    ----
        - call the pad_tensor

    Assert
    ------
        - value error is raised

    '''
    assume(padding_mode not in ALLOWED_PADDING_MODALITIES)
    with pytest.raises(ValueError):
        val = get_padding_values_for_strides((0., 0., 0), (0., 0., 0.), (0., 0., 0.), padding=padding_mode)


#
# Test the actual padding function
#

@given(st.lists(st.integers(128, 256), min_size=2, max_size=3),
       st.lists(st.integers(8, 16), min_size=2, max_size=3),
       st.lists(st.integers(32, 64), min_size=2, max_size=3))
def test_get_valid_padding_values_for_strides_shape_le_image_shape(image_shape, strides, patch_size):
    '''
    The valid padding values should keep the image as the same (if compatible) or smaller than
    the original one.
    Therefore I will check that the founded padding values are lower of equal of the whole image shape 
    Given
    -----
    image_shape: List[int]
        image shape
    strides: List[int]
        Stride dimensions as a List of ints
    patch_size: List[int]
        List of int of the patch size
    Then
    ----
    call _get_valid_padding_values_for_strides

    Assert
    ------
    upper index along each dimension is lower or equal to the image shape along this dimension
    '''
    assume((len(image_shape) == len(patch_size)) & (len(image_shape) == len(strides)))
    values = _get_valid_padding_values_for_strides(array_shape=image_shape, strides=strides, patch_size=patch_size)
    # get the upper values
    upper_values = [x[1] for x in values]
    upper_values.pop(_CHANNEL_AXIS[_DATA_FORMAT])
    upper_values = np.asarray(upper_values)
    assert all(upper_values <= image_shape)



def test_get_same_padding_values_for_strides_correct_vals():
    '''
    Simple unit test to check if the _get_same_padding_values_for_strides
    return the correct padding values.

    Given
    -----
    image_shape = (512, 512, 512)
    strides = (64, 64, 64)
    patch_size = (128, 128, 128)

    Assert
    ------
    padding values equal to (32, 32, 32) in each direction
    '''

    pad_values = _get_same_padding_values_for_strides(array_shape=(512, 512, 512), strides=(64, 64, 64), patch_size=(128, 128, 128))

    # prepare the ground truth.
    # consider also the channel axis
    gt = [[32, 32], [32, 32], [32, 32]]

    if _DATA_FORMAT == 'channels_last':
        gt.append([0, 0])
    else:
        gt.insert(0, [0, 0])

        assert np.all(pad_values == gt)

@pytest.mark.skipif(_FRAMEWORK_NAME == 'torch', reason='test specific for tf.keras or keras framework')
@given(st.tuples(st.integers(16, 64), st.integers(128, 160), st.integers(2,5)),
       st.tuples(st.integers(16, 32), st.integers(32, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_same_correct_pad_2d_tf(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for tf.keras, keras frameworks. 
    Given an image in the format  (h, w, n_channels) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (h, w, n_channels), with n_channels != h != w
        - padding values for each direction (p_h, p_w, 0); p_h != p_w
    Then
    ----
        - pad the image image
    Assert
    ------
        - paddes image shape == (h + p_h, w + p_w, n_channels) 
    '''
    tensor = B.random.normal(t_shape)

    pad_shape = [[0, s] for s in pad_shape]
    pad_shape.append([0, 0])

    padded = _pad_tensor_same(tensor, pad_shape)
    
    gt = tuple([s + w[1] for s, w in zip(t_shape, pad_shape)])

    assert tuple(padded.shape) == gt


@pytest.mark.skipif(_FRAMEWORK_NAME == 'torch', reason='test specific for tf.keras or keras framework')
@given(st.tuples(st.integers(16, 64), st.integers(128, 160), st.integers(256, 288), st.integers(2,5)),
       st.tuples(st.integers(16, 32), st.integers(32, 64), st.integers(21, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_same_correct_pad_3d_tf(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for tf.keras, keras frameworks. 
    Given an image in the format  (h, w, d, n_channels) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (h, w, d, n_channels), with n_channels != h != w != d
        - padding values for each direction (p_h, p_w, 0); p_h != p_w != p_d
    Then
    ----
        - pad the image
    Assert
    ------
        - paddes image shape == (h + p_h, w + p_w, d + p_d, n_channels) 
    
    '''

    tensor = B.random.normal(t_shape)

    pad_shape = [(0, s) for s in pad_shape]
    pad_shape.append((0, 0))

    padded = _pad_tensor_same(tensor, pad_shape)
    
    gt = tuple([s + w[1] for s, w in zip(t_shape, pad_shape)])

    assert tuple(padded.shape) == gt



@pytest.mark.skipif(_FRAMEWORK_NAME != 'torch', reason='test specific fortorch framework')
@given(st.tuples(st.integers(2,5), st.integers(16, 64), st.integers(128, 160)),
       st.tuples(st.integers(16, 32), st.integers(32, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_same_correct_pad_2d_tc(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for torch. 
    Given an image in the format  (n_channels, h, w) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (n_channels, h, w), with n_channels != h != w
        - padding values for each direction (p_h, p_w, 0); p_h != p_w
    Then
    ----
        - pad the image image
    Assert
    ------
        - paddes image shape == (n_channels, h + p_h, w + p_w) 
    '''
    tensor = B.rand(t_shape)

    pad_shape = [(0, s) for s in pad_shape]
    pad_shape.insert(0, (0, 0))

    # remember that pytorch requires tha padding backward than tensorflow
    padded = _pad_tensor_same(tensor, tuple(pad_shape))
    
    gt = tuple([s + w[1] for s, w in zip(t_shape, pad_shape)])

    assert tuple(padded.shape) == gt


@pytest.mark.skipif(_FRAMEWORK_NAME != 'torch', reason='test specific for torch')
@given(st.tuples(st.integers(2,5), st.integers(16, 64), st.integers(128, 160), st.integers(256, 288)),
       st.tuples(st.integers(16, 32), st.integers(32, 64), st.integers(21, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_same_correct_pad_3d_tc(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for torch frameworks. 
    Given an image in the format  (n_channels, h, w, d) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (n_channels, h, w, d, ), with n_channels != h != w != d
        - padding values for each direction (p_h, p_w, 0); p_h != p_w != p_d
    Then
    ----
        - pad the image
    Assert
    ------
        - paddes image shape == (n_channels, h + p_h, w + p_w, d + p_d) 
    
    '''

    tensor = B.rand(t_shape)

    # here the padding shape is formatted as torch requires
    pad_shape = [[0, s] for s in  pad_shape]
    pad_shape.insert(0, [0, 0])

    # remember that pytorch requires tha padding backward than tensorflow
    padded = _pad_tensor_same(tensor, tuple(pad_shape))
    
    gt = tuple([s + w[1] for s, w in zip(t_shape, pad_shape)])

    assert tuple(padded.shape) == gt




@pytest.mark.skipif(_FRAMEWORK_NAME == 'torch', reason='test specific for tf.keras or keras framework')
@given(st.tuples(st.integers(32, 64), st.integers(128, 160), st.integers(2,5)),
       st.tuples(st.integers(8, 16), st.integers(32, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_valid_correct_pad_2d_tf(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for tf.keras, keras frameworks. 
    Given an image in the format  (h, w, n_channels) and a series of padding values along each dimension 
    produce the correctly padding tensor (in valid flavour). 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (h, w, n_channels), with n_channels != h != w
        - padding values for each direction (p_h, p_w, 0); p_h != p_w
    Then
    ----
        - pad the image image
    Assert
    ------
        - paddes image shape == (p_h, p_w, n_channels) 
    '''
    tensor = B.random.normal(t_shape)

    pad_shape = [[0, s] for s in pad_shape]
    pad_shape.append([None, None])

    padded = _pad_tensor_valid(tensor, pad_shape)
    
    gt = [w[1] for w in pad_shape[:-1]]
    gt.append(t_shape[-1])

    assert tuple(padded.shape) == tuple(gt)



@pytest.mark.skipif(_FRAMEWORK_NAME == 'torch', reason='test specific for tf.keras or keras framework')
@given(st.tuples(st.integers(32, 64), st.integers(128, 160), st.integers(256, 288), st.integers(2,5)),
       st.tuples(st.integers(8, 16), st.integers(32, 64), st.integers(21, 64)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_valid_correct_pad_3d_tf(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for tf.keras, keras frameworks. 
    Given an image in the format  (h, w, d, n_channels) and a series of padding values along each dimension 
    produce the correctly padding tensor (in valid flavour). 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (h, w, d, n_channels), with n_channels != h != w
        - padding values for each direction (p_h, p_w, p_d); p_h != p_w != p_d
    Then
    ----
        - pad the image
    Assert
    ------
        - paddes image shape == (p_h, p_w, p_d, n_channels) 
    '''
    tensor = B.random.normal(t_shape)

    pad_shape = [[0, s] for s in pad_shape]
    pad_shape.append([None, None])

    padded = _pad_tensor_valid(tensor, pad_shape)
    
    gt = [w[1] for w in pad_shape[:-1]]
    gt.append(t_shape[-1])

    assert tuple(padded.shape) == tuple(gt)





@pytest.mark.skipif(_FRAMEWORK_NAME != 'torch', reason='test specific fortorch framework')
@given(st.tuples(st.integers(2, 5), st.integers(64, 128), st.integers(128, 160)),
       st.tuples(st.integers(8, 16), st.integers(16, 32)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_valid_correct_pad_2d_tc(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for torch. 
    Given an image in the format  (n_channels, h, w) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (n_channels, h, w), with n_channels != h != w
        - padding values for each direction (p_h, p_w, 0); p_h != p_w
    Then
    ----
        - pad the image image
    Assert
    ------
        - paddes image shape == (n_channels, p_h, p_w) 
    '''
    tensor = B.rand(t_shape)

    pad_shape = [(0, s) for s in pad_shape]
    pad_shape.insert(0, [None, None])

    # remember that pytorch requires tha padding backward than tensorflow
    padded = _pad_tensor_valid(tensor, tuple(pad_shape))

    gt = [w[1] for w in pad_shape[1:]]
    gt.insert(0, t_shape[0])

    assert tuple(padded.shape) == tuple(gt)


@pytest.mark.skipif(_FRAMEWORK_NAME != 'torch', reason='test specific for torch')
@given(st.tuples(st.integers(2,5), st.integers(16, 64), st.integers(128, 160), st.integers(256, 288)),
       st.tuples(st.integers(8, 16), st.integers(32, 64), st.integers(128, 155)))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_padding_valid_correct_pad_3d_tc(t_shape, pad_shape):
    '''
    Check that a correct padding is achieved for torch frameworks. 
    Given an image in the format  (n_channels, h, w, d) and a series of padding values along each dimension 
    produce the correctly padding tensor. 
    This is aims to test that the padding function is stable to framework changes.
    
    Given
    -----
        - tensor image of shape (n_channels, h, w, d, ), with n_channels != h != w != d
        - padding values for each direction (p_h, p_w, 0); p_h != p_w != p_d
    Then
    ----
        - pad the image
    Assert
    ------
        - paddes image shape == (n_channels, h + p_h, w + p_w, d + p_d) 
    
    '''

    tensor = B.rand(t_shape)

    # here the padding shape is formatted as torch requires
    pad_shape = [[0, s] for s in  pad_shape]
    pad_shape.insert(0, [None, None])

    # remember that pytorch requires tha padding backward than tensorflow
    padded = _pad_tensor_valid(tensor, pad_shape)
    
    gt = [w[1] for w in pad_shape[1:]]
    gt.insert(0, t_shape[0])

    assert tuple(padded.shape) == tuple(gt)



