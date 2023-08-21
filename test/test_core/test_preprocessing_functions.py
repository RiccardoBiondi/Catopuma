import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

import numpy as np

from catopuma.core._preprocessing_functions import standard_scaler
from catopuma.core._preprocessing_functions import robust_scaler
from catopuma.core._preprocessing_functions import min_max_scaler
from catopuma.core._preprocessing_functions import identity


@given(st.integers(1, 8), st.integers(1, 4))
def test_identity_return_the_same_image(batch_size: int, channels: int):
    '''
    Test the correct working of the indentity standardization function.

    Given:
        - batch size
        - number of channels
    Then:
        - generate a random image
        - call the identity standardizer
    Assert:
        - the returne image is equal to the input one
    '''

    image = np.random.rand(batch_size, 64, 64, channels)

    std = identity(image=image)
    
    assert np.all(std == image)


@given(st.integers(1, 8), st.integers(1, 4))
def test_min_max_scaler_min_max(batch_size: int, channels: int):
    '''
    test that the image resulting form the min_max_scaler (on the whole batch and channels)
    have min close to 0. and max close to 1..
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the min_max_scaler
    Assert:
        - standardized image min close to 0.
        - standardized image max close to 1.
    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = min_max_scaler(image)


    assert np.isclose(std.min(), 0.)
    assert np.isclose(std.max(), 1.)


@given(st.integers(2, 8), st.integers(1, 4))
def test_min_max_scaler_min_max_per_image(batch_size: int, channels: int):
    '''
    test that the image resulting form the min_max_scaler (image-wise on the whole channels)
    have min close to 0. and max close to 1. for each image
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the min_max_scaler image-wise
    Assert:
        - standardized image min close to 0.
        - standardized image max close to 1.
    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = min_max_scaler(image, axis=(1, 2, 3))


    assert np.all(np.isclose(std.min(axis=(1, 2, 3)), 0.))
    assert np.all(np.isclose(std.max(axis=(1, 2, 3)), 1.))


@given(st.integers(1, 8), st.integers(2, 4))
def test_min_max_scaler_min_max_per_channel(batch_size: int, channels: int):
    '''
    test that the image resulting form min_max_scaler (channel-wise on the whole batch)
    have min close to 0. and max close to 1. for each channel
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the min_max_scaler channel-wise
    Assert:
        - standardized image min close to 0.
        - standardized image max close to 1.
    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = min_max_scaler(image, axis=(0, 1, 2))


    assert np.all(np.isclose(std.min(axis=(0, 1, 2)), 0.))
    assert np.all(np.isclose(std.max(axis=(0, 1, 2)), 1.))


@given(st.integers(2, 8), st.integers(2, 4))
def test_min_max_scaler_min_max_per_image_per_channel(batch_size: int, channels: int):
    '''
    test that the image resulting form min_max_scaler (channel-wise and image-wise)
    have min close to 0. and max close to 1. for each channel and for each image
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the min_max_scaler image-wise and channel-wise
    Assert:
        - standardized image min close to 0.
        - standardized image max close to 1.
    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = min_max_scaler(image, axis=(1, 2))


    assert np.all(np.isclose(std.min(axis=(1, 2)), 0.))
    assert np.all(np.isclose(std.max(axis=(1, 2)), 1.))


@given(st.integers(1, 8), st.integers(1, 4))
def test_standard_scaler_mean_std(batch_size: int, channels: int):
    '''
    test that the image resulting form the standard scaling (on the whole batch and channels)
    have mean close to 0. and standard deviation close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the standard_scaler
    Assert:
        - standardized image mean close to 0.
        - standardized image standard deviation close to 1.

    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = standard_scaler(image)


    assert np.isclose(std.mean(), 0.)
    assert np.isclose(std.std(), 1.)

@given(st.integers(2, 8), st.integers(1, 4))
def test_standard_scaler_mean_std_per_image(batch_size: int, channels: int):
    '''
    test that the image resulting form the standard scaler (image-wise on the whole channels)
    have mean close to 0. and standard deviation close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call standard_scaler image-wise
    Assert:
        - standardized image mean close to 0.
        - standardized image standard deviation close to 1.

    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = standard_scaler(image, axis=(1, 2, 3))


    assert np.all(np.isclose(std.mean(axis=(1, 2, 3)), 0.))
    assert np.all(np.isclose(std.std(axis=(1, 2, 3)), 1.))


@given(st.integers(1, 8), st.integers(2, 4))
def test_standard_scaler_mean_std_per_channel(batch_size: int, channels: int):
    '''
    test that the image resulting form the standard scaler (channel-wise on the whole batch)
    have mean close to 0. and standard deviation close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the standard_scaler channel-wise
    Assert:
        - standardized image mean close to 0.
        - standardized image standard deviation close to 1.

    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = standard_scaler(image, axis=(0, 1, 2))


    assert np.all(np.isclose(std.mean(axis=(0, 1, 2)), 0.))
    assert np.all(np.isclose(std.std(axis=(0, 1, 2)), 1.))


@given(st.integers(2, 8), st.integers(2, 4))
def test_standard_scaler_per_image_per_channel(batch_size: int, channels: int):
    '''
    test that the image resulting form the standard scaler (channel-wise and image-wise)
    havemean close to 0. and standard deviation close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the standard_scaler image-wise and channel-wise
    Assert:
        - standardized image mean close to 0.
        - standardized image standard deviation close to 1.
    '''
    

    # generate an image with both negatie and positive values
    image = 255 * np.random.rand(batch_size, 64, 64, channels) - 127

    std = standard_scaler(image, axis=(1, 2))


    assert np.all(np.isclose(std.mean(axis=(1, 2)), 0.))
    assert np.all(np.isclose(std.std(axis=(1, 2)), 1.))


@given(st.integers(2, 8), st.integers(1, 5))
def test_robust_scaler(batch_size: int, n_channels: int):
    '''
    Test that image resulting form the robust scaler have median close to 0. and iqr close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the robust scaler
    Assert:
        - standardized image median close to 0.
        - standardized image iqr close to 1.
    '''
    image = 255. * np.random.rand(batch_size, 64, 64, n_channels) - 127.

    scaled = robust_scaler(image)

    median = np.median(scaled)
    iqr = np.subtract(*np.percentile(scaled, [75, 25]))
                      
    assert np.isclose(median, 0.)
    assert np.isclose(iqr, 1.)


@given(st.integers(2, 8), st.integers(1, 5))
def test_robust_scaler_per_image(batch_size: int, n_channels: int):
    '''
    Test that image resulting form the robust scaler (image-wise)
    have median close to 0. and iqr close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the robust scaler image wise
    Assert:
        - standardized image median close to 0.
        - standardized image iqr close to 1.
    '''
    
    image = 255. * np.random.rand(batch_size, 64, 64, n_channels) - 127.

    scaled = robust_scaler(image, axis=(1, 2, 3))

    median = np.median(scaled, axis=(1, 2, 3))
    iqr = np.subtract(*np.percentile(scaled, [75, 25], axis=(1, 2, 3)))
                      
    assert np.all(np.isclose(median, 0.))
    assert np.all(np.isclose(iqr, 1.))



@given(st.integers(2, 8), st.integers(1, 5))
def test_robust_scaler_per_channel(batch_size: int, n_channels: int):
    '''
    Test that image resulting form the robust scaler (channel-wise)
    have median close to 0. and iqr close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the robust scaler channel wise
    Assert:
        - standardized image median close to 0.
        - standardized image iqr close to 1.
    '''

    image = 255. * np.random.rand(batch_size, 64, 64, n_channels) - 127.

    scaled = robust_scaler(image, axis=(0, 1, 2))

    median = np.median(scaled, axis=(0, 1, 2))
    iqr = np.subtract(*np.percentile(scaled, [75, 25], axis=(0, 1, 2)))
                      
    assert np.all(np.isclose(median, 0.))
    assert np.all(np.isclose(iqr, 1.))


@given(st.integers(2, 8), st.integers(1, 5))
def test_robust_scaler_per_image_per_channel(batch_size: int, n_channels: int):
    '''
    Test that image resulting form the robust scaler (channel-wise, image-wise)
    have median close to 0. and iqr close to 1.
    I have used close to insetad of == since I have to deal with floating point values.
    
    Given:
        - batch size
        - number of channels
    Then:
        - generate random image
        - call the robust scaler channel wise and image wise
    Assert:
        - standardized image median close to 0.
        - standardized image iqr close to 1.
    '''

    image = 255. * np.random.rand(batch_size, 64, 64, n_channels) - 127.

    scaled = robust_scaler(image, axis=(1, 2))

    median = np.median(scaled, axis=(1, 2))
    iqr = np.subtract(*np.percentile(scaled, [75, 25], axis=(1, 2)))
                      
    assert np.all(np.isclose(median, 0.))
    assert np.all(np.isclose(iqr, 1.))
