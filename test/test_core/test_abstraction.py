'''
'''
import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume

import numpy as np
from typing import Optional

from catopuma.core.abstraction import BaseLoss

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']



legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)
#
# Testing the base loss class and all its operations
#

# Define a mocking class to get rid of the abstract class BaseLoss


class MockLoss(BaseLoss):
    '''
    Simple mocking class to concretize and test the BaseLoss functionality.
    It took an extra argument "const" which allows to control the 
    number returned by the __call__ method and therefore test the aritmtic 
    operations between the losses 
    '''

    def __init__(self, name: Optional[str] = None, const: int = 0):
        
        super().__init__(name=name)

        self.const = const
    
    def __call__(self, y_true, y_pred) -> float:

        return self.const
    

#
# Now start the actual testing
#

def test_base_loss_default_init():
    '''
    Test that the name property is correctly default initialized

    Given:
        - no parameter is required
    Then:
        - init MockLoss object
    Assert:
        - name property is correctly setted as default value
    '''

    loss = MockLoss()

    assert loss.name == 'MockLoss'
    assert loss.__name__ == 'MockLoss'
    assert loss._name == 'MockLoss'


@given(text_strategy)
def test_base_loss_init(name: str):
    '''
    Check if the class name initialization works as expected

    Given:
        - random text as loss name
    Then:
        - init the MockLoss with the custom name
    Assert:
        - name property is correctly assigned to the provided value

    '''

    loss = MockLoss(name=name)


    assert loss.name == name
    assert loss.__name__ == name
    assert loss._name == name


@given(text_strategy)
def test_base_loss_name_setter(name: str):
    '''
    Check that the name setter and getter works as expected

    Given:
        - random text as loss name
    Then:
        - init the MockLoss with default arguments
        - set the name parameter to the provided random name
    Assert:
        - name is "MockLoss" before calling the setter
        - name is the custom one after the setter is called

    '''
    loss = MockLoss()

    assert loss.name == 'MockLoss'
    assert loss.__name__ == 'MockLoss'
    assert loss._name == 'MockLoss'

    loss.name = name

    assert loss.name == name
    assert loss.__name__ == name
    assert loss._name == name

@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_sum_name_and_result(loss1_value: float, loss2_value: float):
    '''
    Check that the sum of two losses provides the correct result when called.
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - loss2 value
    Then:
        - init first loss s.t. return loss1 value
        - init second loss s.t. return loss12 value
        - add the two losses
        - call the resulting loss
    Assert:
        - resulting loss result is loss1_value + loss2_value
        - resulting loss name is loss1_plus_loss2 
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    loss2 = MockLoss(name='loss2', const=loss2_value)

    res = loss1 + loss2
    res_value = res([], [])

    assert res.name == 'loss1_plus_loss2'
    assert np.isclose(res_value, loss1_value + loss2_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_constant_sum_name_and_result(loss1_value: float, const_value: float):
    '''
    Check that the sum of a loss and a constant return the correct value
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - constant value
    Then:
        - init first loss s.t. return loss1 value
        - sum loss and constant
        - call the resulting loss
    Assert:
        - resulting loss result is const_value +const_value
        - resulting loss name is loss1_plus_const_value
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = loss1 + const_value
    res_value = res([], [])

    assert res.name == f'loss1_plus_{const_value}'
    assert np.isclose(res_value, loss1_value + const_value)



@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_subtraction_name_and_result(loss1_value: float, loss2_value: float):
    '''
    Check that the subtraction of two losses provides the correct result when called.
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - loss2 value
    Then:
        - init first loss s.t. return loss1 value
        - init second loss s.t. return loss12 value
        - subtract the two losses
        - call the resulting loss
    Assert:
        - resulting loss result is loss1_value - loss2_value
        - resulting loss name is loss1_minus_loss2 
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    loss2 = MockLoss(name='loss2', const=loss2_value)

    res = loss1 - loss2
    res_value = res([], [])

    assert res.name == 'loss1_minus_loss2'
    assert np.isclose(res_value, loss1_value - loss2_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_constant_subtraction_name_and_result(loss1_value: float, const_value: float):
    '''
    Check that the subtraction of a loss and a constant return the correct value
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - constant value
    Then:
        - init first loss s.t. return loss1 value
        - subtract loss and constant
        - call the resulting loss
    Assert:
        - resulting loss result is const_value - const_value
        - resulting loss name is loss1_minus_const_value
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = loss1 - const_value
    res_value = res([], [])

    assert res.name == f'loss1_minus_{const_value}'
    assert np.isclose(res_value, loss1_value - const_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_multiplication_name_and_result(loss1_value: float, loss2_value: float):
    '''
    Check that the multiplication of two losses provides the correct result when called.
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - loss2 value
    Then:
        - init first loss s.t. return loss1 value
        - init second loss s.t. return loss12 value
        - multiply the two losses
        - call the resulting loss
    Assert:
        - resulting loss result is loss1_value * loss2_value
        - resulting loss name is (loss1)_times_(loss2) 
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    loss2 = MockLoss(name='loss2', const=loss2_value)

    res = loss1 * loss2
    res_value = res([], [])

    assert res.name == '(loss1)_times_(loss2)'
    assert np.isclose(res_value, loss1_value * loss2_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_constant_multiplication_name_and_result(loss1_value: float, const_value: float):
    '''
    Check that the multiplication of a loss and a constant return the correct value
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - constant value
    Then:
        - init first loss s.t. return loss1 value
        - multiply loss and constant
        - call the resulting loss
    Assert:
        - resulting loss result is const_value * const_value
        - resulting loss name is const_value_times_(loss1)
    '''

    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = loss1 * const_value
    res_value = res([], [])

    assert res.name == f'{const_value}_times_loss1'
    assert np.isclose(res_value, loss1_value * const_value)



@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_division_name_and_result(loss1_value: float, loss2_value: float):
    '''
    Check that the division of two losses provides the correct result when called.
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value 
        - loss2 value (different from zero)
    Then:
        - init first loss s.t. return loss1 value
        - init second loss s.t. return loss12 value
        - divide the two losses
        - call the resulting loss
    Assert:
        - resulting loss result is loss1_value / loss2_value
        - resulting loss name is (loss1)_divided_by_(loss2) 
    '''
    assume(loss2_value != 0.)
    loss1 = MockLoss(name='loss1', const=loss1_value)
    loss2 = MockLoss(name='loss2', const=loss2_value)

    res = loss1 / loss2
    res_value = res([], [])

    assert res.name == '(loss1)_divided_by_(loss2)'
    assert np.isclose(res_value, loss1_value / loss2_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_constant_division_name_and_result(loss1_value: float, const_value: float):
    '''
    Check that the division of a loss and a constant return the correct value
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - constant value (different from zero)
    Then:
        - init first loss s.t. return loss1 value
        - multiply loss and constant
        - call the resulting loss
    Assert:
        - resulting loss result is const_value / const_value
        - resulting loss name is (loss1)_divided_by_const_value
    '''
    assume(const_value != 0.)
    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = loss1 / const_value
    res_value = res([], [])

    assert res.name == f'(loss1)_divided_by_{const_value}'
    assert np.isclose(res_value, loss1_value / const_value)



@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_power_name_and_result(loss1_value: float, loss2_value: float):
    '''
    Check that the power of two losses provides the correct result when called.
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value 
        - loss2 value 
    Then:
        - init first loss s.t. return loss1 value
        - init second loss s.t. return loss12 value
        - compute the power of the first loss elevated at the second loss
        - call the resulting loss
    Assert:
        - resulting loss result is loss1_value**loss2_value
        - resulting loss name is (loss1)^(loss2) 
    '''
    assume(loss2_value != 0.)
    loss1 = MockLoss(name='loss1', const=loss1_value)
    loss2 = MockLoss(name='loss2', const=loss2_value)

    res = loss1**loss2
    res_value = res([], [])

    assert res.name == '(loss1)^(loss2)'
    assert np.isclose(res_value, loss1_value**loss2_value)


@given(st.floats(0, 100), st.floats(0, 100))
def test_loss_constant_pow_name_and_result(loss1_value: float, const_value: float):
    '''
    Check that the power of a loss elevated by a constant return the correct value
    Check also that te resulting loss name is valid.

    Given:
        - loss1 value
        - constant value
    Then:
        - init first loss s.t. return loss1 value
        - compute the power of loss elevated by constant
        - call the resulting loss
    Assert:
        - resulting loss result is const_value**const_value
        - resulting loss name is (loss1)^const_value
    '''
    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = loss1**const_value
    res_value = res([], [])

    assert res.name == f'(loss1)^{const_value}'
    assert np.isclose(res_value, loss1_value**const_value)



@given(st.floats(-50., 50),)
def test_loss_neg_name_and_result(loss1_value: float):
    '''
    Check that the negate loss procide the expected result

    Given:
        - loss1 value
    Then:
        - init loss s.t. return loss1 value
        - negate the loss
        - call the negated loss
    Assert:
        - resulting loss value has the same absolute value as loss1_value
        - the ration between the resulting loss value and loss1_value is -1.
        - the loss name is "-1.0_times_loss1"
    '''
    assume(loss1_value != 0.)
    loss1 = MockLoss(name='loss1', const=loss1_value)
    
    res = -loss1
    res_value = res([], [])

    assert res.name == f'-1.0_times_loss1'
    assert np.isclose(abs(res_value), abs(loss1_value))
    assert np.isclose(res_value / loss1_value, -1.)