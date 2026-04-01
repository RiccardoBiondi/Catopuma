import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
from  hypothesis import HealthCheck as HC


import numpy as np

from catopuma.core._preprocessing_functions import standard_scaler
from catopuma.core._preprocessing_functions import robust_scaler
from catopuma.core._preprocessing_functions import min_max_scaler
from catopuma.core._preprocessing_functions import identity
from catopuma.core._preprocessing_functions import unpack_labels

ALLOWED_DATA_FORMATS =   ('channels_first', 'channels_last')

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
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
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
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
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
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
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
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
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


@given(st.integers(2, 8), st.integers(2, 10))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_binary_image_channels_last(batch_size: int, label_value: int):
    '''
    Test that the image returned by the unpack image has only 0. and 1. values

    Given:
        - batch size
        - label to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_last')
    Assert:
        - image unique values are only 0 and 1.
    '''
    image = (label_value * (np.random.rand(batch_size, 64, 64, 1))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_last', labels=[label_value - 1])

    uniques = np.unique(y)

    assert len(uniques) == 2
    assert 0 in uniques
    assert 1 in uniques


@given(st.integers(2, 8), st.integers(2, 10))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_binary_image_channels_first(batch_size: int, label_value: int):
    '''
    Test that the image returned by the unpack image has only 0. and 1. values

    Given:
        - batch size
        - label to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_first')
    Assert:
        - image unique values are only 0 and 1.
    '''
    image = (label_value * (np.random.rand(batch_size, 1, 64, 64))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_first', labels=[label_value - 1])

    uniques = np.unique(y)

    assert len(uniques) == 2
    assert 0 in uniques
    assert 1 in uniques

@given(st.integers(2, 8), st.integers(1, 5))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_correct_shape_channels_last(batch_size: int, n_labels: int):
    '''
    Test that the image returned by the unpack image has a shape equal to the selected number of labels

    Given:
        - batch size
        - number of labels to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_last')
    Assert:
        - the result image shape is (batch_size, 64, 64, number of labels)
    '''
    image = (n_labels * (np.random.rand(batch_size, 64, 64, 1))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_last', labels=list(range(0, n_labels)))

    assert y.shape == (batch_size, 64, 64, n_labels)


@given(st.integers(2, 8), st.integers(1, 5))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_correct_shape_channels_first(batch_size: int, n_labels: int):
    '''
    Test that the image returned by the unpack image has a shape equal to the selected number of labels

    Given:
        - batch size
        - number of labels to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_first')
    Assert:
        - the result image shape is (batch_size, number of labels, 64, 64)
    '''
    image = (n_labels * (np.random.rand(batch_size, 1, 64, 64))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_first', labels=list(range(0,     n_labels)))

    assert y.shape == (batch_size, n_labels, 64, 64)


@given(st.integers(2, 8), st.integers(1, 5))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_correct_label_channels_last(batch_size: int, n_labels: int):
    '''
    Test that each channel of the resulting image is equal to the correct object

    Given:
        - batch size
        - number of labels to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_last')
    Assert:
        - Test that each channel correspond to the correct label
    '''

    image = (n_labels * (np.random.rand(batch_size, 64, 64, 1))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_last', labels=list(range(0, n_labels)))


    for i in range(0, n_labels):

        assert np.all(y[..., i] == (image == i).astype(np.uint8).reshape(batch_size, 64, 64))

@given(st.integers(2, 8), st.integers(1, 5))
@settings(max_examples=5,
        deadline=None,
        suppress_health_check=(HC.too_slow,))
def test_unpack_labels_return_correct_label_channels_first(batch_size: int, n_labels: int):
    '''
    Test that each channel of the resulting image is equal to the correct object

    Given:
        - batch size
        - number of labels to select
    Then:
        - generate random labelmap image
        - call the unpack_labels (data_format='channels_first')
    Assert:
        - Test that each channel correspond to the correct label
    '''

    image = (n_labels * (np.random.rand(batch_size, 1, 64, 64))).astype(np.uint8)

    y = unpack_labels(image, data_format='channels_first', labels=list(range(0, n_labels)))


    for i in range(0, n_labels):

        assert np.all(y[:, i] == (image == i).astype(np.uint8).reshape(batch_size, 64, 64))