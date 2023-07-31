#from catopuma.preprocessing import *
#from catopuma.tensorflow.feeder import *
#from catopuma.losses import *
#from catopuma.uploader import *
import os
#import importlib
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

#spam_loader = importlib.find_loader('spam')
#found = spam_loader is not None

"""
def  set_supported_frameworks() -> None:
    '''
    Function to set the catopuma.core.framework._SUPPORTED_FRAMEWORKS according to 
    the installed libraries; i.e. check if tensorflow, keras and pytorch are available.
    '''

    sup_frameworks = []

    try: 
        import keras
        sup_frameworks.append(fw._KERAS_FRAMEWORK_NAME)
    except:
        pass

    try: 
        import tensorflow.keras
        sup_frameworks.append(fw._TF_KERAS_FRAMEWORK_NAME)
    except:
        pass

    #try: 
    import torch
    #    sup_frameworks.append(fw._TORCH_FRAMEWORK_NAME)
    #except:
    #    pass

    fw._SUPPORTED_FRAMEWORKS = sup_frameworks
"""

# now set the frameworks
#set_supported_frameworks()
_framework = os.environ.get('CATOPUMA_FRAMEWORK', fw._DEFAULT_FRAMEWORK_NAME)
set_framework(_framework)