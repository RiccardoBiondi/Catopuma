'''
Core module to manage the framework settings. Up to now the available frameworks are keras and tensorflow.keras;
but I have planned also to add torch, in order to have a repo to rule them all.
'''
from typing import Tuple




_KERAS_FRAMEWORK_NAME: str = 'keras'
_TF_KERAS_FRAMEWORK_NAME: str = 'tf.keras'

_SUPPORTED_FRAMEWORKS: Tuple[str]  = [_KERAS_FRAMEWORK_NAME, _TF_KERAS_FRAMEWORK_NAME]

global _FRAMEWORK_BACKEND, _FRAMEWORK, _FRAMEWORK_NAME

_FRAMEWORK_NAME = None
_FRAMEWORK_BACKEND = None
_FRAMEWORK = None

_DEFAULT_FRAMEWORK_NAME: str = _KERAS_FRAMEWORK_NAME