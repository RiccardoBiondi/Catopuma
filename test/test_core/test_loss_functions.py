'''
Simple testing module to check if the provided loss functions behave as expected and provides correct results.
'''

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

#import test.custom_strategies as cst


import numpy as np
import tensorflow as tf
import catopuma
from catopuma.core.framework import _FRAMEWORK_BACKEND as K

from catopuma.core._loss_functions import f_score

ALLOWED_DATA_FORMATS =   ('channels_first', 'channels_last')

@given(st.integers(3, 5), st.floats(1., 2.), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans())
@settings(max_examples=10, deadline=None)
def test_f_score_is_in_0_1(n_channels, beta, smooth, data_format, per_image):
    '''
    Test that the f_score is ALWAISE in [0., 1.] indepentently to the 
    provided arguments and computation modalities.

    Given
        - number of image channels
        - beta
        - smoothing factor
        - gathing channel list
        - valid data format
        - per_image modality
    
    Then:
        - generate random gt image batch
        - generate random pr image
        - compute the f_score
    Assert:
        - f_score is higher or equal 0.
        - f_score is lower of  equal 1.
    '''

    y_true = np.random.randint(2, size=(8, 64, 64, n_channels))
    y_true = tf.convert_to_tensor(y_true)
    y_true = K.cast(y_true, 'float32')
    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = K.cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = K.permute_dimensions(y_true, (3, 0, 1, 2))
        y_pred = K.permute_dimensions(y_pred, (3, 0, 1, 2))

    result = f_score(y_pred=y_pred, y_true=y_true, beta=beta, smooth=smooth, data_format=data_format, per_image=per_image) 
    
    assert result >= 0.
    assert result <= 1.


@given(st.integers(2, 16), st.integers(1, 4))
def test_f_score_all_zero_is_zero(batch_size, n_channels):
    '''
    Test that the f_score b=1. (i.e. dice loss) is zero when a zero target image is passed.

    Given
        - batch_size
        - number of channels
    Then
        - generate a zero target and prediction image
        - compute the f_score
    Assert
        - f_score is 0.
    '''

    img = np.zeros((batch_size, 64, 64, n_channels))
    img = img.astype(np.float32) 
    img = tf.convert_to_tensor(img)

    res = f_score(img, img)

    np.isclose(res, 0.)