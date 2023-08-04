'''
Core module to manage the framework settings. Up to now the available frameworks are keras and tensorflow.keras;
but I have planned also to add torch, in order to have a repo to rule them all.
The framework is set by the user and it is used to import the correct backend module.
The backend module is then used to perform the actual computation.

Global variables
----------------

    _FRAMEWORK_BACKEND: ModuleType = None 
            Backend to use for the computation of metrics.
    _FRAMEWORK: ModuleType = None 
            Current framework module
    _FRAMEWORK_NAME: str = None 
            Name of the current framework
    _SUPPORTED_FRAMEWORKS: Iterable[str] = ['keras', 'tf.keras', 'torch'] 
            List of the framework supported by catopuma
    _AVAILABLE_FRAMEWORKS: Iterable[str]  
            List of the framework availble in the current python environment.

Constant
--------

    _KERAS_FRAMEWORK_NAME: str = 'keras'
            Name of the keras framework
    _TF_KERAS_FRAMEWORK_NAME: str = 'tf.keras'
            Name of tensrflow.keras framework
    _TORCH_FRAMEWORK_NAME: str = 'torch'
            Name of pytorch framework
    _FRAMEWORK_LUT: Dict[str, str]
        Look up table mapping the framework name to the correspondig imported module 

Functions
---------
    _retrieve_available_frameworks:
        function to retrieve which framework ara availble in your system
'''
import sys
from typing import Iterable, Dict
# TODO this will be deprecated from python 3.12
from importlib import find_loader


__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']




global _FRAMEWORK_BACKEND, _FRAMEWORK, _FRAMEWORK_NAME, _SUPPORTED_FRAMEWORKS, _AVAILABLE_FRAMEWORKS

# First of all define the framework names.
_KERAS_FRAMEWORK_NAME: str = 'keras'
_TF_KERAS_FRAMEWORK_NAME: str = 'tf.keras'
_TORCH_FRAMEWORK_NAME: str = 'torch'


_FRAMEWORK_LUT: Dict[str, str] = {
        _KERAS_FRAMEWORK_NAME : 'keras',
        _TF_KERAS_FRAMEWORK_NAME : 'tensorflow.keras',
        _TORCH_FRAMEWORK_NAME: 'torch'
    }


_AVAILABLE_FRAMEWORKS: Iterable[str]  = [_KERAS_FRAMEWORK_NAME]


_FRAMEWORK_NAME = None
_FRAMEWORK_BACKEND = None
_FRAMEWORK = None

_DEFAULT_FRAMEWORK_NAME: str = _KERAS_FRAMEWORK_NAME



def _retrieve_available_frameworks() -> Iterable[str]:
    '''
    Search in sys.module which framework are present and return the list
    of the available framework names.
    '''
    # TODO make it raise an exception if no framework is found
    return [k for  k, v in _FRAMEWORK_LUT.items() if find_loader(v) is not None]