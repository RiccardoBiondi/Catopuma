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
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == 'channels_last'


@given(text_strategy, st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(), st.integers(6, 10), st.lists(st.integers(min_value=0, max_value=5,)), st.floats(1e-7, 1e-1))
def test_dice_loss_init(loss_name, data_format, per_image, n_channels, indexes, smooth):
    '''
    Test that the dice loss is correctly initialized with custom parameters

    Given:
        - valid random name
        - valid data format
        - valid weight list
        - valid class indexes
        - valid smoothing factor
        - per image flag
    Then:
        - init dice loss
    
    Assert:
        - correct parameter initialization
    '''

    weigths = np.random.rand(n_channels)

    loss = losses.DiceLoss(name=loss_name, data_format=data_format, per_image=per_image, class_indexes=indexes, class_weights=weigths, smooth=smooth)
    

    assert loss.per_image is per_image
    assert loss.name == loss_name
    assert loss.data_format == data_format
    assert loss.class_indexes == indexes
    assert np.all(np.isclose(loss.class_weights, weigths))
    assert np.isclose(loss.smooth, smooth)