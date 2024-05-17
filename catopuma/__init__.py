'''
'''
import os
from catopuma.__version__ import __version__
import catopuma.core.__framework as fw
from typing import Iterable

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']

# Define some usefull functions to get the availble frameworks, get the current framework 
# and set the current framework

def available_frameworks() -> Iterable[str]:
    '''
    Get the list of availble catopuma frameworks.

    Returns
    -------
    list of str
        list of available frameworks
    '''
    
    return fw._AVAILABLE_FRAMEWORKS


def framework() -> str:
    '''
    Return the current framework
    '''
    return fw._FRAMEWORK_NAME


def set_data_format(data_format: str) -> None:
    '''
    Set the data format to the specified value. 
    Before setting it will check that the data format is supported.
    Supported data formats are(case sentive): 'channels_first' and 'channels_last'.

    Parameter
    ---------
    data_format: str
        string specifying the desidered data format. Could be 'channels_first' or 'channels_last'.

    Raise
    -----
    ValueError
        raise a value error if try to set a non-supported data format. The check is case sensitive.
    '''


    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError(f'{data_format} is not supported. Allowed data formats are: channels_first of channels_last')
    
    fw._DATA_FORMAT = data_format


def data_format() -> str:
    '''
    Return the data format currently in use.
    '''

    return fw._DATA_FORMAT

def set_framework(name: str) -> None:
    '''
    Set the framework to use as backend.
    If a framework is already set, it will be removed and the new one will be set.

    Parameters
    ----------
    name: str
        fremework to use. Can be:`` keras`` or ``tf.keras``
    
    Raises
    ------
        ValueError: in case of incorrect framework name.
        ImportError: in case framework is not installed.
    '''

    if name not in fw._AVAILABLE_FRAMEWORKS:
        raise ValueError(f'The specified framework {name} is not available. Available frameworks are: {fw._AVAILABLE_FRAMEWORKS}')
    
    fw._FRAMEWORK_NAME = name

    # Then import the requested framework
    if fw._FRAMEWORK_NAME == fw._KERAS_FRAMEWORK_NAME:

        import keras
        import tensorflow as tf

        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
        fw._FRAMEWORK_BASE = tf
        set_data_format('channels_last')
    
    elif fw._FRAMEWORK_NAME == fw._TF_KERAS_FRAMEWORK_NAME:

        import tensorflow as tf
        import tensorflow.keras as keras

        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
        fw._FRAMEWORK_BASE = tf
        set_data_format( 'channels_last')


    elif fw._FRAMEWORK_NAME == fw._TORCH_FRAMEWORK_NAME:

        import torch

        fw._FRAMEWORK_BACKEND = torch
        fw._FRAMEWORK = torch
        fw._FRAMEWORK_BASE = torch
        set_data_format('channels_first')


# now set the frameworks
fw._AVAILABLE_FRAMEWORKS = fw._retrieve_available_frameworks()
# TODO fid a better way to manage the default framework
_framework = os.environ.get('CATOPUMA_FRAMEWORK', fw._DEFAULT_FRAMEWORK_NAME)
set_framework(_framework)



# now import the catopuma modules

# import the catopuma agnostic libraries


from catopuma import uploader
from catopuma import preprocessing

# Now import the framework specific libraries.
# No sanithy check is required since I have made them before

if framework() in [fw._TF_KERAS_FRAMEWORK_NAME, fw._KERAS_FRAMEWORK_NAME]:
    from .tensorflow import feeder
    from .tensorflow import losses
else:
    from .torch import feeder
    from .torch import losses