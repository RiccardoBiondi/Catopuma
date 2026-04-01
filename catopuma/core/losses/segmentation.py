from .base import BaseLoss



class DiceLoss(BaseLoss):

    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        
        axes = self._get_reduction_axes(y_true.ndim)

        tp =self.be.sum(y_true * y_pred, axis=axes)
        fp = self.be.sum(y_pred, axis=axes) - tp
        fn = self.be.sum(y_true, axis=axes) - tp

        score = (2 * tp + self.smooth) /  ( 2 * tp + fn + fp + self.smooth)
        
        score = self.mean(score)

        return score