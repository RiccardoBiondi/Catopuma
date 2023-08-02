'''
Core module to manage the framework settings. Up to now the available frameworks are keras and tensorflow.keras;
but I have planned also to add torch, in order to have a repo to rule them all.
The framework is set by the user and it is used to import the correct backend module.
The backend module is then used to perform the actual computation.
'''
import sys
from typing import Iterable, Dict

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']


_KERAS_FRAMEWORK_NAME: str = 'keras'
_TF_KERAS_FRAMEWORK_NAME: str = 'tf.keras'
_TORCH_FRAMEWORK_NAME: str = 'torch'


_FRAMEWORK_LUT: Dict[str, str] = {
        _KERAS_FRAMEWORK_NAME : 'keras',
        _TF_KERAS_FRAMEWORK_NAME : 'tensorflow.keras',
        _TORCH_FRAMEWORK_NAME: 'torch'
    }


global _FRAMEWORK_BACKEND, _FRAMEWORK, _FRAMEWORK_NAME, _SUPPORTED_FRAMEWORKS
_AVAILABLE_FRAMEWORKS: Iterable[str]  = [_KERAS_FRAMEWORK_NAME]


_FRAMEWORK_NAME = None
_FRAMEWORK_BACKEND = None
_FRAMEWORK = None

_DEFAULT_FRAMEWORK_NAME: str = _KERAS_FRAMEWORK_NAME#AVAILBLE_FRAMEWORKS[0]


def clear_framework(framework_name: str): 
    '''
    Clear the framework settings and remove the backend module from the sys.modules.

    Parameters
    ----------
    framework_name: str
        name of the framework to clear. Can be: ``keras``, ``tf.keras`` or ``torch``
    '''
    package_name = _FRAMEWORK_LUT[framework_name]
    loaded_package_modules = [module_name for module_name in sys.modules.keys() if package_name in module_name]
    for module_name in loaded_package_modules:
        sys.modules.pop(module_name, None)
    _FRAMEWORK_NAME = None