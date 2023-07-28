#from catopuma.preprocessing import *
#from catopuma.tensorflow.feeder import *
#from catopuma.losses import *
#from catopuma.uploader import *
import os
from catopuma.__version__ import __version__
import catopuma.core.framework as fw
from typing import Tuple

__author__ = ['Riccardo Biondi']
__email__ = ['riccardo.biondi7@unibo.it']



# now start the injection of the backend, to support both torch and keras (both keraas and tf.keras) implementations



def framework() -> str:
    '''
    Return the current framework
    '''
    return fw._FRAMEWORK_NAME


def set_framework(name: str) -> None:
    '''
    Set the framework to use as backwend.

    Parameters
    ----------
    name: str
        fremework to use. Can be:`` keras`` or ``tf.keras``
    
    Raises
    ------
        ValueError: in case of incorrect framework name.
        ImportError: in case framework is not installed.
    '''

    if name not in fw._SUPPORTED_FRAMEWORKS:
        raise ValueError(f'The specified framework {name} is not valid. Allowed frameworks are: {fw._SUPPORTED_FRAMEWORKS}')
    
    
    fw._FRAMEWORK_NAME = name

    if fw._FRAMEWORK_NAME == fw._KERAS_FRAMEWORK_NAME:
        import keras

        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
    
    elif fw._FRAMEWORK_NAME == fw._TF_KERAS_FRAMEWORK_NAME:

        import tensorflow.keras as keras

        fw._FRAMEWORK_BACKEND = keras.backend
        fw._FRAMEWORK = keras
    
    
 
_framework = os.environ.get('SM_FRAMEWORK', fw._DEFAULT_FRAMEWORK_NAME)
set_framework(_framework)

print(framework())