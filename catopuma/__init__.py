'''
'''
import os
import sys
import importlib
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
    
    elif fw._FRAMEWORK_NAME == fw._TF_KERAS_FRAMEWORK_NAME:

        import tensorflow as tf
        import tensorflow.keras as keras
        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
        fw._FRAMEWORK_BASE = tf

    elif fw._FRAMEWORK_NAME == fw._TORCH_FRAMEWORK_NAME:

        import torch
        fw._FRAMEWORK_BACKEND = torch
        fw._FRAMEWORK = torch
        fw._FRAMEWORK_BASE = torch

# now set the frameworks
fw._AVAILABLE_FRAMEWORKS = fw._retrieve_available_frameworks()
# TODO fid a better way to manage the default framework
_framework = os.environ.get('CATOPUMA_FRAMEWORK', fw._DEFAULT_FRAMEWORK_NAME)
set_framework(_framework)



# now import the catopuma modules

# import the catopuma agnostic libraries


from catopuma import preprocessing
from catopuma import uploader

if framework() in [fw._TF_KERAS_FRAMEWORK_NAME, fw._KERAS_FRAMEWORK_NAME]:
    from .tensorflow import feeder
    from .tensorflow import losses


#__all__ = ['preprocessing', 'uploader', 'feeder', 'framework', '__version__']