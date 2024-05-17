'''
'''


import numpy as np
from typing import Union, Optional, List

import catopuma
from catopuma.core.base_losses import BaseLoss
from catopuma.core._score_functions import f_score
from catopuma.core._score_functions import tversky_score

from catopuma.core.__framework import _FRAMEWORK as F



class DiceScore(BaseLoss, F.nn.Module):
    '''
    '''

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)


class FScore(BaseLoss, F.nn.Module):
    '''
    '''

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)


class JaccardScore(BaseLoss, F.nn.Module):
    '''
    '''

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)


class Precision(BaseLoss, F.nn.Module):

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)


class Recall(BaseLoss, F.nn.Module):

    def __init__(self, name: Optional[str] = None) -> None:

        super().__init__(name=name)


class MatthewsCorrelationCoefficient(BaseLoss, F.nn.Module):

    def __init__(self, name: Optional[str] = None) -> None:

        super().__init__(name=name)