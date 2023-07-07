'''
Testing module to ensure that the loss classes works as expected.
Notice that here it is nottested the computation result since it is verified in  ./test/test_core/test_loss_functions.py testing module.
Here is tested the correct class behaviour
'''


import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

import numpy as np
# import the class to test
from catopuma.losses import DiceLoss



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

    loss = DiceLoss()

    assert loss.name == 'DiceLoss'
    assert np.isclose(loss.smooth, 1e-5) 
    assert loss.per_image is False
    assert loss.class_weights == 1.
    assert loss.class_indexes is  None
    assert loss.data_format == 'channels_last'

