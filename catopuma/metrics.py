from catopuma.core.abstraction import BaseLoss


class JaccardIndex(BaseLoss):
    '''
    '''

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
        return super().__call__(y_true, y_pred)