import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

import numpy as np
import tensorflow.keras.backend as K

# import functions to test
from catopuma.core.functions import _get_required_axis
from catopuma.core.functions import _gather_channels
from catopuma.core.functions import get_reduce_axes
from catopuma.core.functions import gather_channels
from catopuma.core.functions import average

legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

def test_get_required_axis_default():
    '''
    Given: 
        - No parameter required
    Then:
        - call _get_required_axis with default arguments
    Assert:
        - return None
    '''

    required_axis  = _get_required_axis()
    
    assert required_axis is None


def test_get_required_axis_per_image():
    '''
    Given: 
        - No parameter required
    Then:
        - call _get_required_axis with per_image=True
    Assert:
        - required axis is tuple
        - len of the tuple is 3
        - the axis are 1 , 2 and 3 (in this order)
        - the axis are integer types
    '''

    required_axis = _get_required_axis(per_image=True)
    
    assert isinstance(required_axis, tuple)
    assert len(required_axis) == 3
    assert isinstance(required_axis[0], int) & isinstance(required_axis[1], int) & isinstance(required_axis[2], int)
    assert (required_axis[0] == 1) & (required_axis[1] == 2) & (required_axis[2] == 3)

def test_get_required_axis_per_channel():
    '''
    Given: 
        - No parameter required
    Then:
        - call _get_required_axis with per_channel=True
    Assert:
        - required axis is tuple
        - len of the tuple is 3
        - the axis are 0, 1 and 2
        - the axis are integer types
    '''

    required_axis = _get_required_axis(per_channel=True)
    
    assert isinstance(required_axis, tuple)
    assert len(required_axis) == 3
    assert isinstance(required_axis[0], int) & isinstance(required_axis[1], int) & isinstance(required_axis[2], int)
    assert (required_axis[0] == 0) & (required_axis[1] == 1) & (required_axis[2] == 2)


def test_get_required_axis_per_image_per_channel():
    '''
    Given: 
        - No parameter required
    Then:
        - call _get_required_axis with per_image=True, per_channel=True
    Assert:
        - required axis is tuple
        - len of the tuple is 2
        - the axis are 1 and 2 (in this order)
        - the axis are integer types
    '''

    required_axis = _get_required_axis(per_image=True, per_channel=True)
    
    assert isinstance(required_axis, tuple)
    assert len(required_axis) == 2
    assert isinstance(required_axis[0], int) & isinstance(required_axis[1], int)
    assert (required_axis[0] == 1) & (required_axis[1] == 2)


@given(text_strategy)
def test__gather_channel_raise_value_error(data_format: str) :
    '''
    Test the function _gather_channel rainse a ValueError when the wron 
    datafromat is speccified

    Given:
        - data_format str not matching the functoin requirements
    Then:
        - construct a dummy tensor and a dummy index to call the function
        - call the function _gater_channel
    Assert:
        value error is raised
    '''

    dummy_tensor = np.zeros(shape=(1, 64, 64, 1), dtype=np.float32)
    dummy_index = (1)

    with pytest.raises(ValueError):

        x = _gather_channels(x=dummy_tensor, indexes=dummy_index, data_format=data_format)


@given(st.integers(min_value=5, max_value=10), st.lists(st.integers(min_value=0, max_value=5,), min_size=1, max_size=3))
@settings(max_examples=10, deadline=None)
def test__gater_channel_channel_first(number_of_channels, indexes):
    '''
    Test if the function correctly gathe a single channel in channel first modality

    Given:
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

    x = np.concatenate([np.full((1, 1, 64, 64), i) for i in range(0, number_of_channels)], axis=1)
    # get only unique indexes
    indexes = tuple(set(indexes))

    
    gater = _gather_channels(x=x, indexes=indexes, data_format='channel_first')
    
    assert gater.shape == (1, len(indexes), 64, 64)
    assert np.all(np.unique(gater) == sorted(indexes))


@given(st.integers(min_value=5, max_value=10), st.lists(st.integers(min_value=0, max_value=5,), min_size=1, max_size=3))
@settings(max_examples=10, deadline=None)
def test__gater_channel_channel_last(number_of_channels, indexes):
    '''
    Test if the function correctly gathe a single channel in channel first modality

    Given:
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

    x = np.concatenate([np.full((1, 64, 64, 1), i) for i in range(0, number_of_channels)], axis=-1)

    # get only unique indexes
    indexes = tuple(set(indexes))


    
    gater = _gather_channels(x=x, indexes=indexes, data_format='channel_last')
    
    assert gater.shape == (1, 64, 64, len(indexes))
    assert np.all(np.unique(gater) == sorted(indexes))


def test_get_reduce_axis_channel_last():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to False.

    Given:
        - No argument is required
    Then:
        - call get_reduce_axes with per_image=False and image data format reuired to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [0, 1, 2]
    '''

    reduce_axes = get_reduce_axes(per_image=False)

    assert np.all([0, 1, 2] == reduce_axes)



def test_get_reduce_axis_channel_last_per_image():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to False.

    Given:
        - No argument is required
    Then:
        - call get_reduce_axes with per_image=True and image data format required to be 
            'channel_last'
    Assert:
        - resulting reduce axes is [1, 2]
    '''
    reduce_axes = get_reduce_axes(per_image=True)

    assert np.all([1, 2] == reduce_axes)


def test_get_reduce_axis_channel_first():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to False.

    Given:
        - No argument is required
    Then:
        - set the image d data format to 'channel_first'
        - call get_reduce_axes with per_image=False and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [0, 2, 3]
    '''
    K.set_image_data_format('channels_first')

    reduce_axes = get_reduce_axes(per_image=False)

    assert np.all([0, 2, 3] == reduce_axes)


def test_get_reduce_axis_channel_first_per_image():
    '''
    Test that the correct reduction axis is returned when the dataformat is channel_last 
    and per_image is set to False.

    Given:
        - No argument is required
    Then:
        - set the image d data format to 'channel_first'
        - call get_reduce_axes with per_image=True and image data format required to be 
            'channel_first'
    Assert:
        - resulting reduce axes is [2, 3]
    '''
    K.set_image_data_format('channels_first')

    reduce_axes = get_reduce_axes(per_image=True)

    assert np.all([2, 3] == reduce_axes)