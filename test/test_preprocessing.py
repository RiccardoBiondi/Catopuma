import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume

import numpy as np

from typing import Tuple, Dict, Callable

from catopuma.core._preprocessing_functions import standard_scaler
from catopuma.core._preprocessing_functions import robust_scaler
from catopuma.core._preprocessing_functions import min_max_scaler
from catopuma.core._preprocessing_functions import identity

from catopuma.preprocessing import PreProcessing
from catopuma.preprocessing import SCALER_LUT
from catopuma.core.__framework import _DATA_FORMAT

legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

ALLOWED_DATA_FORMATS: Tuple[str] = ('channels_last', 'channels_first')
STANDARDIZERS: Tuple[str] = tuple(SCALER_LUT.keys())

def test_pre_processing_dafault_init():
    '''
    Check thet the PreProcessing class is init with the default 
    arguments

    Given:
        - no argument is required
    Then:
        - init the PreProcessing class
    Assert:
        - default parameters are correctly init
    '''

    preprocess = PreProcessing()


    assert preprocess.data_format == _DATA_FORMAT
    assert preprocess.per_channel is False
    assert preprocess.per_image is False
    assert preprocess.standardizer is SCALER_LUT['identity']
    assert preprocess.target_standardizer is SCALER_LUT['identity']
    assert preprocess.standardizer_params == {}
    assert preprocess.target_standardizer_params == {}

@given(
        st.sampled_from(ALLOWED_DATA_FORMATS),
        st.booleans(),
        st.sampled_from(STANDARDIZERS),
        st.booleans())
def test_pre_processing_init(data_format: str, per_channel: bool,
                            standardizer: str, per_image: bool):
    '''
    Check the PreProcessing argument is properly initialized
    with the provided argument

    Given:
        - valid data format
        - per_channel flag
        - valid standrdizer string
        - per_image flag
    '''
    
    preprocess = PreProcessing(
                            data_format=data_format,
                            per_channel=per_channel,
                            per_image=per_image,
                            standardizer=standardizer,
                            target_standardizer=standardizer)
    
    assert preprocess.data_format == data_format
    assert preprocess.per_channel is per_channel
    assert preprocess.per_image is per_image
    assert preprocess.standardizer is SCALER_LUT[standardizer]
    assert preprocess.target_standardizer is SCALER_LUT[standardizer]


@given(text_strategy)
def test_pre_processing_raise_value_error_wrong_data_format(data_format: str):
    '''
    Check ValueError is raised when an invalid data_format is provided.

    Given:
        - random str, different to valid data_format
    Then:
        - init PreProcessing object
    Assert:
        - ValueError is raised
    '''
    assume(data_format not in ALLOWED_DATA_FORMATS)
    

    with pytest.raises(ValueError):
        preprocess = PreProcessing(data_format=data_format)



@given(text_strategy)
def test_pre_processing_raise_value_error_wrong_standardizaton_string(standardizer: str):
    '''
    Check ValueError is raised when an invalid standardizer string is provided.

    Given:
        - random str, different to a valid standardizer
    Then:
        - init PreProcessing object
    Assert:
        - ValueError is raised
    '''

    assume(standardizer not in STANDARDIZERS)

    with pytest.raises(ValueError):
        prerocess = PreProcessing(standardizer=standardizer)