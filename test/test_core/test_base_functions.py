'''
Module to test the core functions of CATOPUMA.
These functions are fremework dependent, therefore they will be tested for each available framework.
'''

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

import numpy as np
import catopuma

# import functions to test
from catopuma.core._base_functions import _gather_channels
from catopuma.core._base_functions import get_reduce_axes
from catopuma.core._base_functions import gather_channels
from catopuma.core._base_functions import average

from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK_BASE as F
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K

__author__ = ['Riccardo Biondi']
legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1, max_size=15)


# usefull functions to test the function behaviour, because this functions act on the tensors

def _to_tensor(x : np.ndarray):
    
    if _FRAMEWORK_NAME == 'torch':
        return F.from_numpy(x)
    return F.convert_to_tensor(x)


@given(text_strategy)
def test__gather_channel_raise_value_error(data_format: str) :
    '''
    Test the function _gather_channel rainse a ValueError when the wron 
    datafromat is speccified

    Given:
        - Testing framework
        - data_format str not matching the functoin requirements
    Then:
        - construct a dummy tensor and a dummy index to call the function
        - call the function _gater_channel
    Assert:
        value error is raised
    '''

    dummy_tensor = _to_tensor(np.zeros(shape=(1, 64, 64, 1), dtype=np.float32))
    dummy_index = (1)

    with pytest.raises(ValueError):

        x = _gather_channels(x=dummy_tensor, indexes=dummy_index, data_format=data_format)


@given(st.integers(min_value=6, max_value=10), st.lists(st.integers(min_value=0, max_value=5,), min_size=1, max_size=3))
@settings(max_examples=10, deadline=None)
def test__gater_channel_channel_first(number_of_channels, indexes):
    '''
    Test if the function correctly gathe a single channel in channel first modality

    Given:
        - Testing framework
        - an int, representing the number of channels. 
        - an list int representing the channels to gate
    Then:
        - create an image with the given number of channel, where each channel as a constant value equal to its index
        - gathe the required channels in channel first data format
    Assert:
        - gathed thensor shape is (batch_size, n_selected_channel, height, width)
        - the resulting unique values are equal to the gathed channel index
    '''
    
    #construct the tensor

    x = _to_tensor(np.concatenate([np.full((1, 1, 64, 64), i) for i in range(0, number_of_channels)], axis=1))
    # get only unique indexes
    indexes = tuple(set(indexes))

    
    gater = _gather_channels(x=x, indexes=indexes, data_format='channels_first')
    
    assert gater.shape == (1, len(indexes), 64, 64)
    assert np.all(np.unique(gater) == sorted(indexes))


@given(st.integers(min_value=6, max_value=10), st.lists(st.integers(min_value=0, max_value=5,), min_size=1, max_size=3))
@settings(max_examples=10, deadline=None)
def test__gater_channel_channel_last(number_of_channels, indexes):
    '''
    Test if the function correctly gathe a single channel in channel first modality

    Given:
        - Testing framework
        - an int, representing the number of channels. 
        - an list int representing the channels to gate
    Then:
        - create an image with the given number of channel, where each channel as a constant value equal to its index
        - gathe the required channels in channel last data format
    Assert:
        - gathed thensor shape is (batch_size, height, width, n_selected_channel)
        - the resulting unique values are equal to the gathed channel index
    '''
    #construct the tensor

    x = _to_tensor(np.concatenate([np.full((1, 64, 64, 1), i) for i in range(0, number_of_channels)], axis=-1))

    # get only unique indexes
    indexes = tuple(set(indexes))


    
    gater = _gather_channels(x=x, indexes=indexes, data_format='channels_last')
    
    assert gater.shape == (1, 64, 64, len(indexes))
    assert np.all(np.unique(gater) == sorted(indexes))


def test_get_reduce_axes_channel_last_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image/per_chennels are set to False. Consider the case of a 2d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=False and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [0, 1, 2, 3]
    '''
    reduce_axes = get_reduce_axes(per_image=False, per_channel=False)

    assert np.all((0, 1, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_last_per_image_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to True. Consider the case of a 2d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [1, 2, 3]
    '''
    reduce_axes = get_reduce_axes(per_image=True)

    assert np.all((1, 2, 3) == reduce_axes)



def test_get_reduce_axes_channel_last_per_channel_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_channel is set to True. Consider the case of a 2d image.

    Given:
        - No parameter required
    Then:

        - call get_reduce_axes with per_channel=True, per_image=False and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [0, 1, 2]
    '''
    reduce_axes = get_reduce_axes(per_channel=True)

    assert np.all((0, 1, 2) == reduce_axes)


def test_get_reduce_axes_channel_last_per_image_per_channel_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last and 
    both per_image and per_channel are set to True. Consider the case of a 2d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=True and image data format required to be
    Assert:
        - resulting reduce axes is [1, 2]
    '''

    reduce_axes = get_reduce_axes(per_image=True, per_channel=True)
    assert np.all((1, 2) == reduce_axes)


def test_get_reduce_axes_channel_first_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image/per_channel are set to False.

    Given:
        - no parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=False and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [0, 1, 2, 3]
    '''
    reduce_axes = get_reduce_axes(data_format='channels_first')

    assert np.all((0, 1, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_first_per_image_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image is set to True, per_channel is set to False. Consider a 2d image case.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=False and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [1, 2, 3]
    '''
    reduce_axes = get_reduce_axes(per_image=True, data_format='channels_first')

    assert np.all((1, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_first_per_channel_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image is set to False, per_channel is set to True. Consider a 2d image case.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=True and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [0, 2, 3]
    '''
    reduce_axes = get_reduce_axes(per_channel=True, data_format='channels_first')

    assert np.all((0, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_first_per_image_per_channel_2d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first
    and both per_image and per_channel are set to True. Consider a 2d image case.
    
    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=True and image data format required to be
            'channel_first'
    Assert:
        - resulting reduce axes is [2, 3]
    '''

    reduce_axes = get_reduce_axes(per_image=True, per_channel=True, data_format='channels_first')
    
    assert np.all((2, 3) == reduce_axes)

@given(text_strategy)
@settings(max_examples=10, deadline=None)
def test_get_reduce_axes_raise_value_error(data_format: str):
    '''
    Test that get_reduce_axes raise ValueError when the dataformat is not valid.

    Given:
        - data_format str not matching the functoin requirements
    Then:
        - call get_reduce_axes with given image data format
    Assert:
        - ValueError is raised
    '''

    with pytest.raises(ValueError):
        reduce_axes = get_reduce_axes(data_format=data_format)




def test_get_reduce_axes_channel_last_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image/per_chennels are set to False. Consider the case of a 3d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=False and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [0, 1, 2, 3, 4]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, per_image=False, per_channel=False)

    assert np.all((0, 1, 2, 3, 4) == reduce_axes)


def test_get_reduce_axes_channel_last_per_image_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to True. Consider the case of a 3d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [1, 2, 3, 4]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, per_image=True)

    assert np.all((1, 2, 3, 4) == reduce_axes)



def test_get_reduce_axes_channel_last_per_channel_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_channel is set to True. Consider the case of a 3d image.

    Given:
        - No parameter required
    Then:

        - call get_reduce_axes with per_channel=True, per_image=False and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [0, 1, 2, 3]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, per_channel=True)

    assert np.all((0, 1, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_last_per_image_per_channel_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last and 
    both per_image and per_channel are set to True. Consider the case of a 3d image.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=True and image data format required to be
    Assert:
        - resulting reduce axes is [1, 2, 3]
    '''

    reduce_axes = get_reduce_axes(tensor_dims=5, per_image=True, per_channel=True)
    assert np.all((1, 2, 3) == reduce_axes)


def test_get_reduce_axes_channel_first_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image/per_channel are set to False. CPmsider the case of a 3d image.

    Given:
        - no parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=False and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [0, 1, 2, 3, 4]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, data_format='channels_first')

    assert np.all((0, 1, 2, 3, 4) == reduce_axes)


def test_get_reduce_axes_channel_first_per_image_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image is set to True, per_channel is set to False. Consider a 3d image case.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=False and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [1, 2, 3, 4]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, per_image=True, data_format='channels_first')

    assert np.all((1, 2, 3, 4) == reduce_axes)


def test_get_reduce_axes_channel_first_per_channel_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first 
    and per_image is set to False, per_channel is set to True. Consider a 3d image case.

    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=False, per_channel=True and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [0, 2, 3, 4]
    '''
    reduce_axes = get_reduce_axes(tensor_dims=5, per_channel=True, data_format='channels_first')

    assert np.all((0, 2, 3, 4) == reduce_axes)


def test_get_reduce_axes_channel_first_per_image_per_channel_3d():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_first
    and both per_image and per_channel are set to True. Consider a 3d image case.
    
    Given:
        - No parameter required
    Then:
        - call get_reduce_axes with per_image=True, per_channel=True and image data format required to be
            'channel_first'
    Assert:
        - resulting reduce axes is [2, 3, 4]
    '''

    reduce_axes = get_reduce_axes(tensor_dims=5, per_image=True, per_channel=True, data_format='channels_first')
    
    assert np.all((2, 3, 4) == reduce_axes)