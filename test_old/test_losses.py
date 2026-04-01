'''
Testing module to ensure that the loss classes works as expected.
Notice that here it is nottested the computation result since it is verified in  ./test/test_core/test_loss_functions.py testing module.
Here is tested the correct class behaviour.

This module test only the loss properties that are not framework dependent.
The framework dependent properties are tested in ./test/test_core/test_score_functions.py togheter with the computation result.
'''


import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

from catopuma.core.__framework import _DATA_FORMAT

import numpy as np
import catopuma
# import the class to test
from catopuma import losses


legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

ALLOWED_DATA_FORMATS = ('channels_first', 'channels_last')


def test_dice_loss_default_init():
    '''
    test that the class is correctly initialized with the dafault arguments
    Given:
        - no argument is required
    Then:
        - init a DiceLoss object
    Assert:
        - DiceLoss.name is equal to DiceLoss
        - all arguments are to their default values
    '''

    loss = losses.DiceLoss()

    assert loss.name == 'DiceLoss'
    assert np.isclose(loss.smooth, 1e-5) 
    assert loss.per_image is False
    assert loss.per_channel is False
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == _DATA_FORMAT


@given(text_strategy, st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(), st.booleans(), st.integers(6, 10), st.lists(st.integers(min_value=0, max_value=5,)), st.floats(1e-7, 1e-1))
def test_dice_loss_init(loss_name, data_format, per_image, per_channel, n_channels, indexes, smooth):
    '''
    Test that the dice loss is correctly initialized with custom parameters

    Given:
        - valid random name
        - valid data format
        - valid weight list
        - valid class indexes
        - valid smoothing factor
        - per image flag
        - per channel flag

    Then:
        - init dice loss
    
    Assert:
        - correct parameter initialization
    '''

    weigths = np.random.rand(n_channels)

    loss = losses.DiceLoss(name=loss_name, data_format=data_format, per_image=per_image, per_channel=per_channel, class_indexes=indexes, class_weights=weigths, smooth=smooth)
    

    assert loss.per_image is per_image
    assert loss.per_channel is per_channel
    assert loss.name == loss_name
    assert loss.data_format == data_format
    assert loss.class_indexes == indexes
    assert np.all(np.isclose(loss.class_weights, weigths))
    assert np.isclose(loss.smooth, smooth)



def test_tversky_loss_default_init():
    '''
    test that the class is correctly initialized with the dafault arguments
    Given:
        - no argument is required
    Then:
        - init a TverskyLoss object
    Assert:
        - TverskyLoss.name is equal to TverskyLoss
        - all arguments are to their default values
    '''

    loss = losses.TverskyLoss()

    assert loss.name == 'TverskyLoss'
    assert np.isclose(loss.smooth, 1e-5) 
    assert np.isclose(loss.alpha, .5)
    assert np.isclose(loss.beta, .5)
    assert loss.per_image is False
    assert loss.per_channel is False
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == _DATA_FORMAT


@given(text_strategy, st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(), st.booleans(), st.floats(1e-3, 1.), st.integers(6, 10), st.lists(st.integers(min_value=0, max_value=5,)), st.floats(1e-7, 1e-1))
def test_tversky_loss_init(loss_name, data_format, per_image, per_channel, alpha, n_channels, indexes, smooth):
    '''
    Test that the tversky loss is correctly initialized with custom parameters

    Given:
        - valid random name
        - valid data format
        - valid weight list
        - valid class indexes
        - valid smoothing factor
        - per image flag
        - per channel flag
        - alpha value
    Then:
        - compute beta as 1. - alpha
        - init tversky loss
    
    Assert:
        - correct parameter initialization
    '''
    beta = 1. - alpha
    weigths = np.random.rand(n_channels)

    loss = losses.TverskyLoss(
                                name=loss_name, data_format=data_format,
                                per_image=per_image, per_channel=per_channel,
                                alpha=alpha, beta=beta,
                                class_indexes=indexes, class_weights=weigths, smooth=smooth)
    

    assert loss.per_image is per_image
    assert loss.per_channel is per_channel
    assert loss.name == loss_name
    assert loss.data_format == data_format
    assert loss.class_indexes == indexes
    assert np.all(np.isclose(loss.class_weights, weigths))
    assert np.isclose(loss.smooth, smooth)
    assert np.isclose(loss.alpha, alpha)
    assert np.isclose(loss.beta, beta)



def test_mean_squared_error_loss_default_init():
    '''
    test that the class is correctly initialized with the dafault arguments
    Given:
        - no argument is required
    Then:
        - init a MeanSquaredError object
    Assert:
        - MeanSquaredError.name is equal to MeanSquaredError
        - all arguments are to their default values
    '''

    loss = losses.MeanSquaredError()

    assert loss.name == 'MeanSquaredError'
    assert loss.per_image is False
    assert loss.per_channel is False
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == _DATA_FORMAT


@given(text_strategy, st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(), st.booleans(), st.integers(6, 10), st.lists(st.integers(min_value=0, max_value=5,)))
def test_mean_squared_error_loss_init(loss_name, data_format, per_image, per_channel, n_channels, indexes):
    '''
    Test that the mean squared error is correctly initialized with custom parameters

    Given:
        - valid random name
        - valid data format
        - valid weight list
        - valid class indexes
        - per image flag
        - per channel flag
    Then:
        - init mean squared error
    
    Assert:
        - correct parameter initialization
    '''
    weigths = np.random.rand(n_channels)

    loss = losses.MeanSquaredError(
                                name=loss_name, data_format=data_format,
                                per_image=per_image, per_channel=per_channel,
                                class_indexes=indexes, class_weights=weigths)
    

    assert loss.name == loss_name
    assert loss.data_format == data_format
    assert loss.class_indexes == indexes
    assert np.all(np.isclose(loss.class_weights, weigths))




def test_mean_absolute_error_loss_default_init():
    '''
    test that the class is correctly initialized with the dafault arguments
    Given:
        - no argument is required
    Then:
        - init a MeanAbsoluteError object
    Assert:
        - MeanAbsoluteError.name is equal to MeanAbsoluteError
        - all arguments are to their default values
    '''

    loss = losses.MeanAbsoluteError()

    assert loss.name == 'MeanAbsoluteError'
    assert loss.per_image is False
    assert loss.per_channel is False
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == _DATA_FORMAT


@given(text_strategy, st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(), st.booleans(), st.integers(6, 10), st.lists(st.integers(min_value=0, max_value=5,)))
def test_mean_absolute_error_loss_init(loss_name, data_format, per_image, per_channel, n_channels, indexes):
    '''
    Test that the mean absolute error is correctly initialized with custom parameters

    Given:
        - valid random name
        - valid data format
        - valid weight list
        - valid class indexes
        - per image flag
        - per channel flag
    Then:
        - init mean absolute error
    
    Assert:
        - correct parameter initialization
    '''
    weigths = np.random.rand(n_channels)

    loss = losses.MeanAbsoluteError(
                                name=loss_name, data_format=data_format,
                                per_image=per_image, per_channel=per_channel,
                                class_indexes=indexes, class_weights=weigths)
    

    assert loss.name == loss_name
    assert loss.data_format == data_format
    assert loss.class_indexes == indexes
    assert np.all(np.isclose(loss.class_weights, weigths))
