import pytest
import hypothesis.strategies as st
from hypothesis import given, assume, settings

import numpy as np

from catopuma.uploader import SimpleITKUploader
from catopuma.feeder import ImageFeederOnTheFly


def test_image_feeder_on_the_fly_default_init():
    '''
    Check if all the dafault parameters are correctly setted

    Given:
        - no argument is required
    Then:
        - init the ImageFeederOnTheFly. Use a placeholder list of path
    Assert:
        - all arguments are equal to the default ones
    '''

    # create the palceholder paths
    img_paths = 8 * ['test/test_images/test_image.nii']
    tar_paths = 8 * ['test/test_images/test_target.nii']

    feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths)

    assert feeder.shuffle is True
    assert feeder.batch_size == 8
    assert isinstance(feeder.uploader, SimpleITKUploader)
    assert feeder.preprocessing is None
    assert feeder.augmentation_strategy is None

    # check indexes is correctly initialized

    assert feeder.indexes.min() == 0
    assert feeder.indexes.max() == 7
    assert len(feeder.indexes) == 8

@given(st.integers(2, 32), st.booleans())
def test_image_feeder_on_the_fly_init(batch_size, shuffle):
    '''
    assert that the init by specifying parameters is correct

    Given:
        - batch_size
        - shuffle flag
    Then:
        - create palceholder paths for target and images
        - init the ImageFeederOnTheFly 
    Assert:
        - all parameters are setted correctly
    '''
    img_paths = batch_size * ['test/test_images/test_image.nii']
    tar_paths = batch_size * ['test/test_images/test_target.nii']

    feeder = ImageFeederOnTheFly(
                                img_paths=img_paths,
                                target_paths=tar_paths, 
                                batch_size=batch_size,
                                shuffle=shuffle)
    
    assert feeder.batch_size == batch_size
    assert feeder.shuffle is shuffle

    # check also that indexes is correctly initialized
    assert feeder.indexes.min() == 0
    assert feeder.indexes.max() == batch_size - 1
    assert len(feeder.indexes) == batch_size

@given(st.integers(1, 100), st.integers(1, 100))
def test_image_feeder_on_the_fly_raise_value_error_path_different_lenght(image_path_size, target_path_size):
    '''
    Chech that the image feeder raise value error when different lenght of
    image and target paths are provided

    Given:
        - lenght of image paths
        - lenght of target paths
    Then:
        - create the image and target paths list
        - init the image feeder
    Assert:
        - value error is raised
    '''
    
    assume(image_path_size != target_path_size)
    img_paths = image_path_size * ['test/test_images/test_image.nii']
    tar_paths = target_path_size * ['test/test_images/test_target.nii']

    with pytest.raises(ValueError):
        feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths)

@given(st.integers(1, 64), st.integers(1, 64))
def test_image_feeder_on_the_fly_images_len_lower_than_batch(path_len, batch_size):
    '''
    Check the image feeder raise value error when the leng of the provide image
    paths is lower than the batch size
    Given:
        - lenght of image and target path
        - batch size > lenght
    Then:
        - crete the dummy path list
        - init the image feeder
    Assert:
        - value error is raised
    '''
    assume(path_len < batch_size)

    img_paths = path_len * ['test/test_images/test_image.nii']
    tar_paths = path_len * ['test/test_images/test_target.nii']


    with pytest.raises(ValueError):
        feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths, batch_size=batch_size)


@given(st.integers(8, 100))
def test_image_feeder_on_the_fly_shuffle_true(path_len):
    '''
    Check that the indexes are randomly shuffled when shuffle is set to True

    Given:
        - image and target path length
    Then:
        - init ImgaeFeederOnTheFly with shuffle option activated
    Assert:
        - indexes has correct max and min (path length - 1, 0)
        - indexes has correct len (path lenght)
        - indexes is different from the ordered list
    '''

    img_paths = path_len * ['test/test_images/test_image.nii']
    tar_paths = path_len * ['test/test_images/test_target.nii']

    feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths, shuffle=True)

    assert feeder.indexes.min() == 0
    assert feeder.indexes.max() == path_len - 1
    assert len(feeder.indexes) == path_len
    assert np.any(feeder.indexes != np.arange(0, path_len, 1))


@given(st.integers(8, 100))
def test_image_feeder_on_the_fly_shuffle_false(path_len):
    '''
    Check that the indexes are not randomly shuffled when shuffle is set to False

    Given:
        - image and target path length
    Then:
        - init ImgaeFeederOnTheFly with shuffle option deactivated
    Assert:
        - indexes has correct max and min (path length - 1, 0)
        - indexes has correct len (path lenght)
        - indexes is equal the ordered list
    '''

    img_paths = path_len * ['test/test_images/test_image.nii']
    tar_paths = path_len * ['test/test_images/test_target.nii']

    feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths, shuffle=False)

    assert feeder.indexes.min() == 0
    assert feeder.indexes.max() == path_len - 1
    assert len(feeder.indexes) == path_len
    assert np.all(feeder.indexes == np.arange(0, path_len, 1))


@given(st.integers(1, 8), st.integers(1, 100))
def test_image_feeder_on_the_fly_len(batch_size, number_of_batches):
    '''
    Check that the len of the feeder (i.e. the number of created batches) is N when 
    the path len is in range [N * batch_size, (N + 1) * batch_size[
    
    Given:
        - batch_size
        - number of batches to create
    Then:
        - generate random noise (to sum to the path len)
        - init the image feeder
    Assert:
        - len of the feeder is equal to the provided number of batches
    '''

    # generate the noise
    noise = int(batch_size * np.random.rand(1))

    number_of_paths = number_of_batches * batch_size + noise

    assume(number_of_paths >= number_of_batches * batch_size)
    assume(number_of_paths < (number_of_batches + 1) * batch_size)

    img_paths = number_of_paths * ['test/test_images/test_image.nii']
    tar_paths = number_of_paths * ['test/test_images/test_target.nii']

    

    feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths, batch_size=batch_size)

    assert len(feeder) == number_of_batches


@given(st.integers(1, 64), st.integers(10, 1000))
@settings(max_examples=10, deadline=None)
def test_each_batch_has_correct_size_and_shape(batch_size, number_of_paths):
    '''
    Check that the returned batches has the correct size and shape
    Given:
        - batch size
        - number of paths
    Then:
        - create the test img and target path list
        - init the ImageFeederOnTheFly
        - get random item
    Assert:
        - len of the item is 2
        - shape of the first element is (batch_size, 64, 64, 1)
        - shape of the seconf element is (batch_size, 64, 64, 1)
    '''

    assume(batch_size <= number_of_paths)

    img_paths = number_of_paths * ['test/test_images/test_image.nii']
    tar_paths = number_of_paths * ['test/test_images/test_target.nii'] 

    feeder = ImageFeederOnTheFly(img_paths=img_paths, target_paths=tar_paths, batch_size=batch_size)
    # get the reference element idex

    idx = np.random.randint(0, len(feeder))

    item = feeder[idx]
    
    assert len(item) == 2
    assert item[0].shape == (batch_size, 64, 64, 1)
    assert item[1].shape == (batch_size, 64, 64, 1)