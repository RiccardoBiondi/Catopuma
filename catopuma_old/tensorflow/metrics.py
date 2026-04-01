'''
'''


import numpy as np
from typing import Union, Optional, List

import catopuma
from catopuma.core.base_losses import BaseLoss
from catopuma.core._score_functions import f_score
from catopuma.core._score_functions import tversky_score


class DiceScore(BaseLoss):
    '''
    '''

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)


class FScore(BaseLoss):
    '''
    '''

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)


class JaccardScore(BaseLoss):
    '''
    '''

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)