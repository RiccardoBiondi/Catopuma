import pytest
import hypothesis.strategies as st
from hypothesis import given, settings


from catopuma.uploader import SimpleITKUploader

ALLOWED_DATA_FORMATS =   ('channels_first', 'channels_last')

legitimate_chars = st.characters(whitelist_categories=('Lu', 'Ll'),
                                 min_codepoint=65, max_codepoint=90)

text_strategy = st.text(alphabet=legitimate_chars, min_size=1,
                        max_size=15)

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

    assert loader.data_format == 'channels_last'


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


def  test_simple_itk_uploader_correct_output_tuple_lenght():
    '''
    Check that the call method return a tuple of len 2, representing the input and 
    the target images

    Given:  
        - No argument is required
    Then:
        - init SimpleITKUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - the returned tuple has len == 2
    '''

    loader = SimpleITKUploader()

    samples = loader('test/test_images/test_image.nii', 'test/test_images/test_target.nii')
    
    assert len(samples) == 2


def test_simple_itk_uploader_channels_last_correct_output_shape():
    '''
    Check that the red images have the correct shape, coherent with the channels_last 
    data format

    Given:
        - no argument is required
    Then:
        - init the  SimpleITKUploader
        - call the uploader providing a valid image and target paths
    Assert:
        - image and target shape are (64, 64, 1)
    '''
    
    loader = SimpleITKUploader()

    X, y = loader('test/test_images/test_image.nii', 'test/test_images/test_target.nii')

    assert X.shape == (64, 64, 1)
    assert y.shape == (64, 64, 1)



def test_simple_itk_uploader_channels_first_correct_output_shape():
    '''
    Check that the red images have the correct shape, coherent with the channels_first 
    data format

    Given:
        - no argument is required
    Then:
        - init the  SimpleITKUploader with channels_first as data_format
        - call the uploader providing a valid image and target paths
    Assert:
        - image and target shape are (1, 64, 64)
    '''
    
    loader = SimpleITKUploader(data_format='channels_first')

    X, y = loader('test/test_images/test_image.nii', 'test/test_images/test_target.nii')

    assert X.shape == (1, 64, 64)
    assert y.shape == (1, 64, 64)
