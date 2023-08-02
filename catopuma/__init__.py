#from catopuma.preprocessing import *
#from catopuma.tensorflow.feeder import *
#from catopuma.losses import *
#from catopuma.uploader import *
import os
import sys
import importlib
from catopuma.__version__ import __version__
import catopuma.core.framework as fw
from typing import Iterable

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']



# now start the injection of the backend, to support both torch and keras (both keraas and tf.keras) implementations

def set_available_frameworks():
    '''
    Retrieve which of the supported frameworks are available in the system.

    '''
    fw._AVAILABLE_FRAMEWORKS = [k for  k, v in fw._FRAMEWORK_LUT.items() if importlib.find_loader(v) is not None]

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
        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
    
    elif fw._FRAMEWORK_NAME == fw._TF_KERAS_FRAMEWORK_NAME:

        import tensorflow.keras as keras
        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras

    elif fw._FRAMEWORK_NAME == fw._TORCH_FRAMEWORK_NAME:

        import torch
        fw._FRAMEWORK_BACKEND = torch
        fw._FRAMEWORK = torch

# now set the frameworks
set_available_frameworks()
_framework = os.environ.get('CATOPUMA_FRAMEWORK', fw._DEFAULT_FRAMEWORK_NAME)
set_framework(_framework)