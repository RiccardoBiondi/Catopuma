from .base import BaseLoss



class DiceLoss(BaseLoss):

    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth