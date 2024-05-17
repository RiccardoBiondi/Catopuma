import catopuma
import catopuma.core.__framework as fw

import pytest
import hypothesis.strategies as st
from hypothesis import given, assume, settings

import numpy as np

from catopuma.uploader import SimpleITKUploader
from catopuma import feeder


@pytest.mark.skipif(fw._FRAMEWORK_NAME not in ['keras', 'tf.keras'], reason="Test only works with tf.keras and keras frameworks")
class TestImageFeederTensorflow:
        
    def test_image_feeder_on_the_fly_default_init(self):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths)
        assert imfeeder.shuffle is True
        assert imfeeder.batch_size == 8
        assert isinstance(imfeeder.uploader, SimpleITKUploader)
        assert imfeeder.preprocessing is None
        assert imfeeder.augmentation_strategy is None
        # check indexes is correctly initialized
        assert imfeeder.indexes.min() == 0
        assert imfeeder.indexes.max() == 7
        assert len(imfeeder.indexes) == 8


    @given(st.integers(2, 32), st.booleans())
    def test_image_feeder_on_the_fly_init(self, batch_size, shuffle):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, batch_size=batch_size, shuffle=shuffle)
        assert imfeeder.batch_size == batch_size
        assert imfeeder.shuffle is shuffle
        # check also that indexes is correctly initialized
        assert imfeeder.indexes.min() == 0
        assert imfeeder.indexes.max() == batch_size - 1
        assert len(imfeeder.indexes) == batch_size


    @given(st.integers(5, 100), st.integers(10, 100))
    def test_image_feeder_on_the_fly_raise_value_error_path_different_lenght(self, image_path_size, target_path_size):
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
        img2_paths = image_path_size * ['test/test_images/test_image.nii']
        tar_paths = target_path_size * ['test/test_images/test_target.nii']

        with pytest.raises(ValueError):
            imfeeder = feeder.ImageFeederOnTheFly(img_paths, img2_paths, tar_paths)


    @given(st.integers(1, 64), st.integers(1, 64))
    def test_image_feeder_on_the_fly_images_len_lower_than_batch(self, path_len, batch_size):
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
            imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, batch_size=batch_size)


    @given(st.integers(8, 100))
    def test_image_feeder_on_the_fly_shuffle_true(self, path_len):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, shuffle=True)
        assert imfeeder.indexes.min() == 0
        assert imfeeder.indexes.max() == path_len - 1
        assert len(imfeeder.indexes) == path_len
        assert np.any(imfeeder.indexes != np.arange(0, path_len, 1))


    @given(st.integers(8, 100))
    def test_image_feeder_on_the_fly_shuffle_false(self, path_len):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, shuffle=False)
    
        assert imfeeder.indexes.min() == 0
        assert imfeeder.indexes.max() == path_len - 1
        assert len(imfeeder.indexes) == path_len
        assert np.all(imfeeder.indexes == np.arange(0, path_len, 1))


    @given(st.integers(1, 8), st.integers(1, 100))
    def test_image_feeder_on_the_fly_len(self, batch_size, number_of_batches):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, batch_size=batch_size)
        assert len(imfeeder) == number_of_batches


    @given(st.integers(1, 64), st.integers(10, 1000))
    @settings(max_examples=10, deadline=None)
    def test_each_batch_has_correct_size_and_shape(self, batch_size, number_of_paths):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths, batch_size=batch_size)
        # get the reference element idex
        idx = np.random.randint(0, len(imfeeder))
        item = imfeeder[idx]
        assert len(item) == 2
        assert item[0].shape == (batch_size, 64, 64, 1)
        assert item[1].shape == (batch_size, 64, 64, 1)


@pytest.mark.skipif(fw._FRAMEWORK_NAME != 'torch', reason="Test only works with torch frameworks")
class TestFeederTorch:

    def test_image_feeder_on_the_fly_default_init(self):
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
        imfeeder = feeder.ImageFeederOnTheFly(img_paths, tar_paths)
        assert isinstance(imfeeder.uploader, SimpleITKUploader)
        assert imfeeder.preprocessing is None
        assert imfeeder.augmentation_strategy is None
        # check indexes is correctly initialized

    @given(st.integers(5, 100), st.integers(10, 100))
    def test_image_feeder_on_the_fly_raise_value_error_path_different_lenght(self, image_path_size, target_path_size):
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
        img2_paths = image_path_size * ['test/test_images/test_image.nii']
        tar_paths = target_path_size * ['test/test_images/test_target.nii']

        with pytest.raises(ValueError):
            imfeeder = feeder.ImageFeederOnTheFly(img_paths, img2_paths, tar_paths)


    @given(st.integers(1, 8), st.integers(1, 100))
    def test_image_feeder_on_the_fly_len(self, n_channels, number_of_images):
        '''
        Check that the len of the feeder (i.e. the number of images in the dataset) is number_of_images when 
        the path len number_of_images.
        
        Given:
            - number_of_images
            - number of image channels
        Then:
            - image_paths
            - init the image feeder
        Assert:
            - len of the feeder is equal to the provided number of batches
        '''
        # generate the noise

        img_paths = number_of_images * ['test/test_images/test_image.nii']
        channels_paths = n_channels * [img_paths]
        tar_paths = number_of_images * ['test/test_images/test_target.nii']
        imfeeder = feeder.ImageFeederOnTheFly(*channels_paths, tar_paths)
        assert len(imfeeder) == number_of_images


    @given(st.integers(1, 5), st.integers(10, 1000))
    @settings(max_examples=10, deadline=None)
    def test_each_batch_has_correct_size_and_shape(self, n_channels, number_of_paths):
        '''
        Check that the returned image has the correct size and shape
        Given:
            - numebr of channels
            - number of paths
        Then:
            - create the test img and target path list
            - init the ImageFeederOnTheFly
            - get random item
        Assert:
            - len of the item is 2
            - shape of the first element is  (n_channels, 64, 64)
            - shape of the seconf element is (1, 64, 64)
        '''

        img_paths = number_of_paths * ['test/test_images/test_image.nii']
        imgs = n_channels * [img_paths]
        tar_paths = number_of_paths * ['test/test_images/test_target.nii'] 
        imfeeder = feeder.ImageFeederOnTheFly(*imgs, tar_paths,)
        # get the reference element idex
        idx = np.random.randint(0, len(imfeeder))
        item = imfeeder[idx]

        assert len(item) == 2
        assert item[0].shape == (n_channels, 64, 64)
        assert item[1].shape == (1, 64, 64)


