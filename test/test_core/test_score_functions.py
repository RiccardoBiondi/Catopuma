'''
Simple testing module to check if the provided loss functions behave as expected and provides correct results.
The loss functions are framework dependent, therefore will be tested for each available framework.
'''


import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume

from itertools import product
import sklearn.metrics as ref_metrics



import numpy as np
#import tensorflow as tf

import catopuma
from catopuma.core.__framework import _FRAMEWORK_NAME
from catopuma.core.__framework import _FRAMEWORK_BASE as F
from catopuma.core.__framework import _FRAMEWORK_BACKEND as K
from catopuma.core.__framework import _DATA_FORMAT
from catopuma.core._score_functions import f_score
from catopuma.core._score_functions import tversky_score
from catopuma.core._score_functions import mse
from catopuma.core._score_functions import mae

ALLOWED_DATA_FORMATS =   ('channels_first', 'channels_last')

# fisrt of all define some helpet functions to manage the tensor depending on the framework

if _FRAMEWORK_NAME == 'torch':
    import torch
    DATA_TYPES = {'float32': torch.float32}
else:
    import tensorflow as tf
    DATA_TYPES = {'float32': 'float32'}



def _to_tensor(x : np.ndarray):
    
    if _FRAMEWORK_NAME == 'torch':
        return F.from_numpy(x)
    return F.convert_to_tensor(x)


def _cast(x, new_type):

    if _FRAMEWORK_NAME == 'torch':
        return x.type(DATA_TYPES[new_type])
    return K.cast(x, DATA_TYPES[new_type])

def _permute_dimensions(x, indexes):

    if _FRAMEWORK_NAME == 'torch':
        return K.permute(x, indexes)
    
    return K.permute_dimensions(x, indexes) 
# start the actual testing


@given(st.integers(3, 5), st.floats(1., 2.), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans())
@settings(max_examples=10, deadline=None)
def test_f_score_is_in_0_1(n_channels, beta, smooth, data_format, per_image):
    '''
    Test that the f_score is ALWAISE in [0., 1.] indepentently to the 
    provided arguments and computation modalities.

    Given:
        - number of image channels
        - beta
        - smoothing factor
        - gathing channel list
        - valid data format
        - per_image modality

    Then:
        - set the framework to the given one
        - generate random gt image batch
        - generate random pr image
        - compute the f_score
    Assert:
        - f_score is higher or equal 0.
        - f_score is lower of  equal 1.
    '''

    y_true = np.random.randint(2, size=(8, 64, 64, n_channels))
    y_true = _to_tensor(y_true)
    y_true = _cast(y_true, 'float32')
    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = _to_tensor(y_pred)
    y_pred = _cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = _permute_dimensions(y_true, (3, 0, 1, 2))
        y_pred = _permute_dimensions(y_pred, (3, 0, 1, 2))

    result = f_score(y_pred=y_pred, y_true=y_true, beta=beta, smooth=smooth, data_format=data_format, per_image=per_image) 

    assert result >= 0.
    assert result <= 1.


@given(st.integers(2, 16), st.integers(1, 4))
@settings(max_examples=10, deadline=None)
def test_f_score_all_zero_is_zero(batch_size, n_channels):
    '''
    Test that the f_score b=1. (i.e. dice loss) is zero when a zero target image is passed.

    Given:
        - batch_size
        - number of channels
    Then:
        - set the framework to the given one
        - generate a zero target and prediction image
        - compute the f_score
    Assert:
        - f_score is 0.
    '''

    img = np.zeros((batch_size, 64, 64, n_channels))
    img = img.astype(np.float32) 
    img = _to_tensor(img)

    res = f_score(img, img)

    np.isclose(res, 0.)


@given(st.integers(1, 16), st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_f_score_beta_1_is_one(batch_size, n_channels):
    '''
    Test the f1_score is close to one when two equal images are provided as prediction and target.

    Given:
        - target batch size
        - target number of channels
    Then:
        - generate random image
        - compute the f1_score with target equal to prediction
    Assert:
        - resulting score is close to one    
    ''' 

    tar = np.random.randint(0, 1, (batch_size, 64, 64, n_channels))
    tar = tar.astype(np.float32)
    tar = _to_tensor(tar)
 

    
    loss = f_score(y_pred=tar, y_true=tar, beta=1.)

    assert np.isclose(1., loss)


@given(st.integers(1, 16), st.integers(1, 5), st.floats(1., 2.))
@settings(max_examples=10, deadline=None)
def test_f_score_is_zero(batch_size, n_channels, beta):
    '''
    Test the loss is close to zero when two equal images are passed as 
    input and target

    Given:
        - target batch size
        - target number of channels
        - beta value
    Then:
        - generate random image
        - init the DiceLoss 
        - compute f_score with target different to prediction
    Assert:
        - resulting loss is close to zero    
    '''

    tar = np.random.randint(0, 1, (batch_size, 64, 64, n_channels))
    tar = tar.astype(np.float32)
    pred = 1. - tar
    tar = _to_tensor(tar)
    pred = _to_tensor(pred)
    
    loss = f_score(y_pred=pred, y_true=tar, beta=beta)

    assert np.isclose(0., loss)
    

#
# Now the tests for the twersky score
#

@given(st.integers(3, 5), st.floats(0., 1.), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans())
@settings(max_examples=10, deadline=None)
def test_tversky_score_is_in_0_1(n_channels, alpha, smooth, data_format, per_image):
    '''
    Test that the tversky_score is ALWAISE in [0., 1.] indepentently to the 
    provided arguments and computation modalities.

    Given:
        - number of image channels
        - alpha
        - smoothing factor
        - gathing channel list
        - valid data format
        - per_image modality

    Then:
        - set the framework to the given one
        - compute beta as 1 - alpha
        - generate random gt image batch
        - generate random pr image
        - compute the tversky_score
    Assert:
        - tversky_score is higher or equal 0.
        - tversky_score is lower of  equal 1.
    '''

    beta = 1. - alpha
    y_true = np.random.randint(2, size=(8, 64, 64, n_channels))
    y_true = _to_tensor(y_true)
    y_true = _cast(y_true, 'float32')
    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = _to_tensor(y_pred)
    y_pred = _cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = _permute_dimensions(y_true, (3, 0, 1, 2))
        y_pred = _permute_dimensions(y_pred, (3, 0, 1, 2))

    result = tversky_score(y_pred=y_pred, y_true=y_true, alpha=alpha, beta=beta, smooth=smooth, data_format=data_format, per_image=per_image) 

    assert result >= 0.
    assert result <= 1.


@given(st.integers(2, 16), st.integers(1, 4), st.floats(0., 1.))
@settings(max_examples=10, deadline=None)
def test_tversky_score_all_zero_is_zero(batch_size, n_channels, alpha):
    '''
    Test that the tversky_score b=1. (i.e. dice loss) is zero when a zero target image is passed.

    Given:
        - batch_size
        - number of channels
        - alpha
    Then:
        - set the framework to the given one
        - compute beta as 1 - alpha
        - generate a zero target and prediction image
        - compute the tversky_score
    Assert:
        - tversky_score is 0.
    '''

    beta = 1. - alpha
    img = np.zeros((batch_size, 64, 64, n_channels))
    img = img.astype(np.float32) 
    img = _to_tensor(img)

    res = tversky_score(y_pred=img, y_true=img, alpha=alpha, beta=beta)

    np.isclose(res, 0.)



@given(st.integers(3, 5), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans())
@settings(max_examples=10, deadline=None)
def test_tversky_score_is_f_score_alpha_beta_05(n_channels, smooth, data_format, per_image):
    '''
    Test that the tversky_score is equal to f1_score (i.e., dice score) if alpha and beta parameters are equal to 0.5

    Given:
        - number of image channels
        - smoothing factor
        - gathing channel list
        - valid data format
        - per_image modality

    Then:
        - generate random gt image batch
        - generate random pr image
        - compute the f1_score
        - compute the twersky score (alpha=beta=0.5)
    Assert:
        - tversky_score is close to f1_score
    '''

    y_true = np.random.randint(2, size=(8, 64, 64, n_channels))
    y_true = _to_tensor(y_true)
    y_true = _cast(y_true, 'float32')
    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = _to_tensor(y_pred)
    y_pred = _cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = _permute_dimensions(y_true, (3, 0, 1, 2))
        y_pred = _permute_dimensions(y_pred, (3, 0, 1, 2))

    dice = f_score(y_pred=y_pred, y_true=y_true, beta=1., smooth=smooth, data_format=data_format, per_image=per_image)
    result = tversky_score(y_pred=y_pred, y_true=y_true, alpha=.5, beta=.5, smooth=smooth, data_format=data_format, per_image=per_image) 

    assert np.isclose(result, dice)


@given(st.integers(3, 5), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(),  st.floats(0.1, 0.9))
@settings(max_examples=10, deadline=None)
def test_f1_score_is_close_to_sklean_result(n_channels, smooth, data_format, per_channel, threshold):
    '''
    Test that the f_score is close to the f_score resulting from sklearn implementation (used as reference)
    
    Given
        - number of image channels
        - smoothing factor
        - gating channel list
        - valid data format (either channels first or channels last)
        - per_channel modality
        - binarization threshold
    Then
        - generate random gt image batch (binary)
        - generate random pr image (binary)
        - compute reference f_score with sklearn
        - compute the f1_score
    Assert
        - reference f_score close to computed one
    '''
    y_true = np.random.randint(2, size=(8, 64, 64, n_channels))
    y_true = (y_true > threshold).astype(np.uint8)

    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = (y_pred > threshold).astype(np.uint8)

    for i in range(n_channels):
        y_pred[..., i] = (i + 1) * y_pred[..., i]
        y_true[..., i] = (i + 1) * y_true[..., i] 

    average = 'macro' if per_channel else 'micro'

    f_score_ref = ref_metrics.f1_score(y_true.flatten(), y_pred.flatten(), average=average, labels=list(range(1, n_channels + 1)))

    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    
    y_true = _to_tensor(y_true)
    y_true = _cast(y_true, 'float32')

    y_pred = _to_tensor(y_pred)
    y_pred = _cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = _permute_dimensions(y_true, (3, 0, 1, 2))
        y_pred = _permute_dimensions(y_pred, (3, 0, 1, 2))
    
    dice = f_score(y_pred=y_pred, y_true=y_true, beta=1., smooth=smooth, data_format=data_format, per_channel=per_channel, per_image=False)
    
    assert np.isclose(dice, f_score_ref, atol=1e-4)



@given(st.integers(3, 5), st.floats(0., 1e-3), st.sampled_from(ALLOWED_DATA_FORMATS), st.booleans(),  st.floats(0.1, 0.99), st.floats(0.1, 0.9))
@settings(max_examples=10, deadline=None)
def test_f_beta_score_is_close_to_sklean_result(n_channels, smooth, data_format, per_channel, beta, threshold):
    '''
    Test that the f_score is close to the f_score resulting from sklearn implementation (used as reference)
    
    Given
        - number of image channels
        - smoothing factor
        - gating channel list
        - valid data format (either channels first or channels last)
        - per_channel flag
        - binarization threshold
    Then
        - generate random gt image batch (binary)
        - generate random pr image (binary)
        - compute reference f_score with sklearn
        - compute the f1_score
    Assert
        - reference f_score close to computed one
    '''

    # now assign a different lable to each channel (to test also the per_channel behaviour)
    # the per_channel = True corresponds to the 'macro' sklearn average modality
    y_true = np.random.randint(2, size=(8, 64, 64, n_channels), dtype=np.uint8)

    y_pred = np.random.rand(8, 64, 64, n_channels)
    y_pred = (y_pred > threshold).astype(np.uint8)

    for i in range(n_channels):
        y_pred[..., i] = (i + 1) * y_pred[..., i]
        y_true[..., i] = (i + 1) * y_true[..., i] 

    average = 'macro' if per_channel else 'micro'

    f_score_ref = ref_metrics.fbeta_score(y_true.flatten(), y_pred.flatten(), beta=beta, average=average, labels=list(range(1, n_channels + 1)))

    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    
    y_true = _to_tensor(y_true)
    y_true = _cast(y_true, 'float32')

    y_pred = _to_tensor(y_pred)
    y_pred = _cast(y_pred, 'float32')

    if data_format == 'channels_first':
        y_true = _permute_dimensions(y_true, (0, 3, 1, 2))
        y_pred = _permute_dimensions(y_pred, (0, 3, 1, 2))
    
    dice = f_score(y_pred=y_pred, y_true=y_true, beta=beta, smooth=smooth, data_format=data_format, per_channel=per_channel, per_image=False)
    
    assert np.isclose(dice, f_score_ref, atol=1e-5)



@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3), st.booleans(), st.booleans())
@settings(max_examples=10, deadline=None)
def test_mse_is_zero_for_equal_imges(hgl, lgl, n_images, n_channels, per_image, per_channel):
    '''
    Test that the Mean Squared Error is close to zero if compute using the same image as ground
    truth and prediction.

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
        - per_image
        - per_channel
    Then
    ----
        - generate the test image
        - compute the mse between the test image and itself
    Assert
    ------
        - mse is close to 0.
    '''
    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    img = _to_tensor(np.random.uniform(lgl, hgl, img_shape).reshape(img_shape))

    mse_val = mse(img, img, per_channel=per_channel, per_image=per_image)

    assert np.isclose(mse_val, 0.)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mse_is_close_to_sklearn_mse(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Squared Error is close to the scikit learn one when called with per_image=False and
    per_channel=False

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mse)
        - compute the mse using the implemented loss (tst_mse)
    Assert
    ------
        - tst_mse is close to ref_mse
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    ref_mse = ref_metrics.mean_squared_error(gt_img.ravel(), pr_img.ravel())

    tst_mse = mse(_to_tensor(gt_img), _to_tensor(pr_img), per_image=False, per_channel=False)

    assert np.isclose(ref_mse, tst_mse)

    

@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mse_is_close_to_sklearn_mse_per_image(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Squared Error is close to the scikit learn one when called with per_image=True and
    per_channel=False

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mse)
        - compute the mse using the implemented loss (tst_mse)
    Assert
    ------
        - tst_mse is close to ref_mse
    '''
    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)

    ref_mses = [ref_metrics.mean_squared_error(gt_img[i].ravel(), pr_img[i].ravel()) for i in range(n_images)]
    
    ref_mse = np.mean(ref_mses)
    tst_mse = mse(_to_tensor(gt_img), _to_tensor(pr_img), per_image=True, per_channel=False)

    assert np.isclose(ref_mse, tst_mse)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mse_is_close_to_sklearn_mse_per_channel(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Squared Error is close to the scikit learn one when called with per_image=False and
    per_channel=True

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mse)
        - compute the mse using the implemented loss (tst_mse)
    Assert
    ------
        - tst_mse is close to ref_mse
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    if _DATA_FORMAT == 'channels_last':

        ref_mses = [ref_metrics.mean_squared_error(gt_img[..., i].ravel(), pr_img[..., i].ravel()) for i in range(n_channels)]
    else:
        ref_mses = [ref_metrics.mean_squared_error(gt_img[:, i].ravel(), pr_img[:, i].ravel()) for i in range(n_channels)]
    ref_mse = np.mean(ref_mses)
    tst_mse = mse(_to_tensor(gt_img), _to_tensor(pr_img), per_image=False, per_channel=True)

    assert np.isclose(ref_mse, tst_mse)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mse_is_close_to_sklearn_mse_per_image_per_channel(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Squared Error is close to the scikit learn one when called with per_image=True and
    per_channel=True

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mse)
        - compute the mse using the implemented loss (tst_mse)
    Assert
    ------
        - tst_mse is close to ref_mse
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    if _DATA_FORMAT == 'channels_last':

        ref_mses = [ref_metrics.mean_squared_error(gt_img[j, ..., i].ravel(), pr_img[j, ..., i].ravel()) for i, j in product(range(n_channels), range(n_images))]
    else:
        ref_mses = [ref_metrics.mean_squared_error(gt_img[j, i].ravel(), pr_img[j, i].ravel()) for i, j in product(range(n_channels), range(n_images))]
    ref_mse = np.mean(ref_mses)
    tst_mse = mse(_to_tensor(gt_img), _to_tensor(pr_img), per_image=True, per_channel=True)

    assert np.isclose(ref_mse, tst_mse)


































@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3), st.booleans(), st.booleans())
@settings(max_examples=10, deadline=None)
def test_mae_is_zero_for_equal_imges(hgl, lgl, n_images, n_channels, per_image, per_channel):
    '''
    Test that the Mean Absolute Error is close to zero if compute using the same image as ground
    truth and prediction.

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
        - per_image
        - per_channel
    Then
    ----
        - generate the test image
        - compute the mae between the test image and itself
    Assert
    ------
        - mae is close to 0.
    '''
    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    img = _to_tensor(np.random.uniform(lgl, hgl, img_shape).reshape(img_shape))

    mse_val = mae(img, img, per_channel=per_channel, per_image=per_image)

    assert np.isclose(mse_val, 0.)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mae_is_close_to_sklearn_mae(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Absolute Error is close to the scikit learn one when called with per_image=False and
    per_channel=False

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mae)
        - compute the mse using the implemented loss (tst_mae)
    Assert
    ------
        - tst_mae is close to ref_mae
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    ref_mae = ref_metrics.mean_absolute_error(gt_img.ravel(), pr_img.ravel())

    tst_mae = mae(_to_tensor(gt_img), _to_tensor(pr_img), per_image=False, per_channel=False)

    assert np.isclose(ref_mae, tst_mae)

    

@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mae_is_close_to_sklearn_mae_per_image(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Absolute Error is close to the scikit learn one when called with per_image=True and
    per_channel=False

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mae using the sklearn function  (ref_mse)
        - compute the mae using the implemented loss (tst_mse)
    Assert
    ------
        - tst_mae is close to ref_mae
    '''
    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)

    ref_maes = [ref_metrics.mean_absolute_error(gt_img[i].ravel(), pr_img[i].ravel()) for i in range(n_images)]
    
    ref_mae = np.mean(ref_maes)
    tst_mae = mae(_to_tensor(gt_img), _to_tensor(pr_img), per_image=True, per_channel=False)

    assert np.isclose(ref_mae, tst_mae)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mse_is_close_to_sklearn_mae_per_channel(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Absolute Error is close to the scikit learn one when called with per_image=False and
    per_channel=True

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mse using the sklearn function  (ref_mae)
        - compute the mse using the implemented loss (tst_mae)
    Assert
    ------
        - tst_mae is close to ref_mae
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    if _DATA_FORMAT == 'channels_last':

        ref_maes = [ref_metrics.mean_absolute_error(gt_img[..., i].ravel(), pr_img[..., i].ravel()) for i in range(n_channels)]
    else:
        ref_maes = [ref_metrics.mean_absolute_error(gt_img[:, i].ravel(), pr_img[:, i].ravel()) for i in range(n_channels)]
    ref_mae = np.mean(ref_maes)
    tst_mae = mae(_to_tensor(gt_img), _to_tensor(pr_img), per_image=False, per_channel=True)

    assert np.isclose(ref_mae, tst_mae)


@given(st.floats(1., 100.), st.floats(-100., 10), st.integers(1, 16), st.integers(1, 3))
@settings(max_examples=10, deadline=None)
def test_mae_is_close_to_sklearn_mae_per_image_per_channel(hgl, lgl, n_images, n_channels):
    '''
    Test that the Mean Absolute Error is close to the scikit learn one when called with per_image=True and
    per_channel=True

    Given
    -----
        - highest gray level
        - lowest gray level
        - number of images
        - number of channels
    Then
    ----
        - generate the test gt image
        - gemerate the test pred image
        - compute the mae using the sklearn function  (ref_mae)
        - compute the mae using the implemented loss (tst_mae)
    Assert
    ------
        - tst_mae is close to ref_mae
    '''

    if _DATA_FORMAT == 'channels_first':
        img_shape = (n_images, n_channels, 128, 128)
    else:
        img_shape = (n_images, 128, 128, n_channels)

    assume(lgl < hgl)
    
    gt_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)
    pr_img = np.random.uniform(lgl, hgl, img_shape).reshape(img_shape)


    if _DATA_FORMAT == 'channels_last':

        ref_maes = [ref_metrics.mean_absolute_error(gt_img[j, ..., i].ravel(), pr_img[j, ..., i].ravel()) for i, j in product(range(n_channels), range(n_images))]
    else:
        ref_maes = [ref_metrics.mean_absolute_error(gt_img[j, i].ravel(), pr_img[j, i].ravel()) for i, j in product(range(n_channels), range(n_images))]
    ref_mae = np.mean(ref_maes)
    tst_mae = mae(_to_tensor(gt_img), _to_tensor(pr_img), per_image=True, per_channel=True)

    assert np.isclose(ref_mae, tst_mae)