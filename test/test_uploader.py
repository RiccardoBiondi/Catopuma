import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume

import numpy as np
from catopuma.uploader import SimpleITKUploader
from catopuma.uploader import LazyPatchBaseUploader

from catopuma.core.__framework import _DATA_FORMAT

ALLOWED_DATA_FORMATS =   ('channels_first', 'channels_last')

legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=90)
text_strategy = st.text(alphabet=legitimate_chars, min_size=1, max_size=15)

def test_simple_itk_uploader_default_init():
    '''
    Test that all the dafault parametner are set correctly

    Given:
        - No argument is required
    Then:
        - init the SimpleITKUploader with default arguments
    Assert:
        all the class members are init to their default values
    '''

    loader = SimpleITKUploader()

    assert loader.data_format == _DATA_FORMAT


@given(st.sampled_from(ALLOWED_DATA_FORMATS))
def test_simple_itk_uploader_init(data_format: str):
    '''
    Check that SimpleITKUploader arguments are correctly instantiated

    Given:
        - valid data format
    Then:
        - init SimpleITKUploader
    Assert:
        - SimpleITKUploader data format is equal to the input one
    '''
    
    loader = SimpleITKUploader(data_format=data_format)

    assert loader.data_format == data_format


@given(text_strategy)
def test_simple_itk_uploader_raise_value_error(data_format: str):
    '''
    Check taht a value error is raised when SimpleITKUploader is 
    init with an unsupported data format

    Given:
        - random strin as data format 
    Then:
        - init SimpleITKUploader with the invalid data format
    Assert:
        - value error is raised
    '''

    with pytest.raises(ValueError):
        loader = SimpleITKUploader(data_format=data_format)

@given(st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def  test_simple_itk_uploader_correct_output_tuple_lenght(n_images):
    '''
    Check that the call method return a tuple of len 2, representing the input and 
    the target images

    Given: 
        - number of input images
    Then:
        - init SimpleITKUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - the returned tuple has len == 2
    '''
    imgs = n_images * ['test/test_images/test_image.nii']
    loader = SimpleITKUploader()

    samples = loader(*imgs, 'test/test_images/test_target.nii')
    
    assert len(samples) == 2

@given(st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_simple_itk_uploader_channels_last_2d__correct_output_shape(n_images):
    '''
    Check that the red images have the correct shape, coherent with the channels_last 
    data format

    Given:
        - number of input images
    Then:
        - init the  SimpleITKUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - image shape is (64, 64, n_images)
        - target shape is (64, 64, 1)
    '''
    imgs = n_images * ['test/test_images/test_image.nii']
    loader = SimpleITKUploader(data_format='channels_last')

    X, y = loader(*imgs, 'test/test_images/test_target.nii')

    assert X.shape == (64, 64, n_images)
    assert y.shape == (64, 64, 1)


@given(st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_simple_itk_uploader_channels_first_2d_correct_output_shape(n_images):
    '''
    Check that the red images have the correct shape, coherent with the channels_first 
    data format

    Given:
        - number of input images
    Then:
        - init the  SimpleITKUploader with channels_first as data_format
        - call the uploader providing a valid image and target paths
    Assert:
        - image is (n_images, 64, 64)
        - target shape is (1, 64, 64)
    '''
    imgs = n_images * ['test/test_images/test_image.nii']
    loader = SimpleITKUploader(data_format='channels_first')

    X, y = loader(*imgs, 'test/test_images/test_target.nii')

    assert X.shape == (n_images, 64, 64)
    assert y.shape == (1, 64, 64)


@given(st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_simple_itk_uploader_channels_last_3d__correct_output_shape(n_volumes):
    '''
    Check that the red volume have the correct shape, coherent with the channels_last 
    data format

    Given:
        - number of input volume
    Then:
        - init the  SimpleITKUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - volume shape is (64, 64, 64, n_volumes)
        - target shape is (64, 64, 64, 1)
    '''
    vols = n_volumes * ['test/test_images/test_volume.nii']
    loader = SimpleITKUploader(data_format='channels_last')

    X, y = loader(*vols, 'test/test_images/test_volume_target.nii')

    assert X.shape == (64, 64, 64, n_volumes)
    assert y.shape == (64, 64, 64, 1)

@given(st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_simple_itk_uploader_channels_first_3d_correct_output_shape(n_volumes):
    '''
    Check that the red volume have the correct shape, coherent with the channels_first 
    data format

    Given:
        - number of input volume
    Then:
        - init the  SimpleITKUploader with channels_first as data_format
        - call the uploader providing a valid image and target paths
    Assert:
        - volume shape is (n_volumes, 64, 64, 64)
        - target shape is (1, 64, 64, 64)
    '''
    vols = n_volumes * ['test/test_images/test_volume.nii']
    loader = SimpleITKUploader(data_format='channels_first')

    X, y = loader(*vols, 'test/test_images/test_volume_target.nii')

    assert X.shape == (n_volumes, 64, 64, 64)
    assert y.shape == (1, 64, 64, 64)


def test_lazy_patch_base_uploader_default_init():
    '''
    Test that all the dafault parametner are set correctly

    Given:
        - No argument is required
    Then:
        - init the LazyPatchBaseUploade with default arguments
    Assert:
        all the class members are init to their default values
    '''

    loader = LazyPatchBaseUploader((16, 16))

    assert loader.data_format == _DATA_FORMAT
    assert loader.threshold == -1


@given(st.sampled_from(ALLOWED_DATA_FORMATS), st.floats(0., 1.), st.lists(st.integers(16, 32), min_size=2, max_size=3 ))
def test_lazy_patch_base_uploader_init(data_format: str, threshold: float, patch_size):
    '''
    Check that LazyPatchBaseUploader arguments are correctly instantiated

    Given:
        - valid data format
        - valid threshold value
        - valid patch size
    Then:
        - init LazyPatchBaseUploader
    Assert:
        - LazyPatchBaseUploader data format is equal to the input one
    '''
    
    loader = LazyPatchBaseUploader(patch_size=patch_size, threshold=threshold, data_format=data_format)

    assert np.all(np.asarray(loader.patch_size) == np.asarray(patch_size))
    assert loader.threshold == threshold
    assert loader.data_format == data_format


@given(text_strategy)
def test_lazy_patch_base_uploader_raise_value_error_for_wrong_data_format(data_format: str):
    '''
    Check taht a value error is raised when LazyPatchBaseUploader is init with an unsupported data format

    Given:
        - random strin as data format 
    Then:
        - init LazyPatchBaseUploader with the invalid data format
    Assert:
        - value error is raised
    '''

    with pytest.raises(ValueError):
        loader = LazyPatchBaseUploader(patch_size=(16, 16), data_format=data_format)

@given(st.integers(2, 3), st.integers(8, 64), st.integers(16, 64))
def test_lazy_patch_base_uploader_check_consistency_patch_outside_image_raise_value_error(dimension, image_size, patch_size):
    '''
    Check that the _checkConsistency method raise ValueError when the required patch dimension 
    is bigger than the input image

    Given:
        - dimension
        - image_size
        - patch_size
    Then:
        - init  LazyPatchBaseUploader
        - call the _checkConsistency method
    Assert:
        - value error is raised
    '''
    assume(image_size < patch_size)
    
    im_size = dimension * [image_size]
    pc_size = dimension * [patch_size]


    loader = LazyPatchBaseUploader(patch_size=pc_size, data_format=_DATA_FORMAT)

    with pytest.raises(ValueError):

        _ = loader._checkConsistency(image_size=im_size)


@given(st.integers(2, 3), st.integers(2, 3))
def test_lazy_patch_base_uploader_check_consistency_mismatch_dimension_raise_value_error(patch_dim, image_dim):
    '''
    Check that the checkConsistency metod raise value error when patch size and image size have diferent dimension

    Given:
        - patch dimension
        - image dimension
    Then:
        - assume patch_dimesion != image_dimension
        - create patch and image sizes
        - init  LazyPatchBaseUploader
        - call the _checkConsistencyMethod
    Assert:
        - _checkConsistency method raise value error
    '''

    assume(patch_dim != image_dim)

    patch_size = patch_dim * [16]
    image_size = image_dim * [64]

    loader = LazyPatchBaseUploader(patch_size=patch_size, data_format=_DATA_FORMAT)

    with pytest.raises(ValueError):
        _ = loader._checkConsistency(image_size=image_size)


@given(st.lists(st.integers(16, 32), min_size=2, max_size=3 ))
def test_lazy_patch_base_uploader_sample_patch_origin_is_zero(patch_size):
    '''
    Check that the sampled patch origin is alwaise zero when the patch_size and the image size are the 
    same.

    Given:
        - patch_size
    Then:
        - Init the LazyPatchBaseUploader
        - call the _samplePatchOrigin method with image size equal to the patch size
    Assert:
        - sampled patch origin is 0
    '''

    loader = LazyPatchBaseUploader(patch_size=patch_size)

    origin = loader._samplePatchOrigin(patch_size)
    
    zeros = len(origin) * [0]

    assert np.all(np.asarray(zeros) == np.asarray(origin))



@given(st.integers(2, 3), st.integers(4, 16), st.integers(32, 128))
def test_lazy_patch_base_uploader_sample_patch_origin_is_in_image(dimensions, patch_size, image_size):
    '''
    Check that the sampled patch origin, and the whole patch, are inside the image

    Given:
        - dimensions 
        - patch_size
        - image_size
    Then:
        - build image and patch sizes
        - Init the LazyPatchBaseUploader
        - call the _samplePatchOrigin
    Assert:
        - patch origin and upper indexes are inside the image
    '''

    patch_size = dimensions * [patch_size]
    image_size = dimensions * [image_size]

    loader = LazyPatchBaseUploader(patch_size=patch_size)

    origin =  loader._samplePatchOrigin(image_size=image_size)
    
    upper_index = np.asarray(origin) + np.asarray(patch_size)
    assert np.all(np.asarray(origin) >= np.zeros(dimensions))
    assert np.all(np.asarray(origin) < np.asarray(image_size))
    assert np.all(np.asarray(upper_index) < np.asarray(image_size))


@given(st.integers(4, 16), st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_lazy_patch_base_uploader_channels_last_2d_correct_output_shape(patch_size, n_images):
    '''
    Check that the red images have the correct shape, coherent with the channels_last 
    data format and the specified patch size

    Given:
        - patch_size
        - number of input images
    Then:
        - init the  LazyPatchBaseUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - image shape is (patch_size, n_images)
        - target shape is (patch_size, 1)
        '''
    imgs = n_images * ['test/test_images/test_image.nii']
    loader = LazyPatchBaseUploader(patch_size=(patch_size, patch_size), data_format='channels_last')

    X, y = loader(*imgs, 'test/test_images/test_target.nii')

    assert X.shape == (patch_size, patch_size, n_images)
    assert y.shape == (patch_size, patch_size, 1)


@given(st.integers(4, 16), st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_lazy_patch_base_uploader_channels_first_2d_correct_output_shape(patch_size,  n_images):
    '''
    Check that the red images have the correct shape, coherent with the channels_first 
    data formatand the specified patch size

    Given:
        - patch_size
        - number of input images
    Then:
        - init the  LazyPatchBaseUploader with channels_first as data_format
        - call the uploader providing a valid image and target paths
    Assert:
        - image shape is (n_images, patch_size)
        - target shape is (1, patch_size)
            '''
    imgs = n_images * ['test/test_images/test_image.nii']
    loader = LazyPatchBaseUploader(patch_size=(patch_size, patch_size), data_format='channels_first')

    X, y = loader(*imgs, 'test/test_images/test_target.nii')

    assert X.shape == (n_images, patch_size, patch_size)
    assert y.shape == (1, patch_size, patch_size)


@given(st.integers(4, 16), st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_lazy_patch_base_uploader_channels_last_3d_correct_output_shape(patch_size, n_volumns):
    '''
    Check that the red volumes have the correct shape, coherent with the channels_last 
    data format and the specified patch size

    Given:
        - patch_size
        - number of input volumes
    Then:
        - init the  LazyPatchBaseUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - image shape is (patch_size, n_volumns)
        - target shape is (patch_size, 1)

        '''
    vols = n_volumns * ['test/test_images/test_volume.nii']
    loader = LazyPatchBaseUploader(patch_size=(patch_size, patch_size, patch_size), data_format='channels_last')

    X, y = loader(*vols, 'test/test_images/test_volume_target.nii')

    assert X.shape == (patch_size, patch_size, patch_size, n_volumns)
    assert y.shape == (patch_size, patch_size, patch_size, 1)


@given(st.integers(4, 16), st.integers(1, 5))
@settings(max_examples=10, deadline=None)
def test_lazy_patch_base_uploader_channels_first_3d_correct_output_shape(patch_size, n_volumns):
    '''
    Check that the red images have the correct shape, coherent with the channels_first 
    data formatand the specified patch size

    Given:
        - patch_size
        - number of input volumes
    Then:
        - init the  LazyPatchBaseUploader with channels_first as data_format
        - call the uploader providing a valid image and target paths
    Assert:
        - image shape is (n_volums, patch_size)
        - target shape is (1, patch_size)
        '''
    vols = n_volumns * ['test/test_images/test_volume.nii']
    loader = LazyPatchBaseUploader(patch_size=(patch_size, patch_size, patch_size), data_format='channels_first')

    X, y = loader(*vols, 'test/test_images/test_volume_target.nii')

    assert X.shape == (n_volumns, patch_size, patch_size, patch_size)
    assert y.shape == (1, patch_size, patch_size, patch_size)