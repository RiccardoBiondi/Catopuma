from typing import Optional
from abc import ABC, abstractmethod
from catopuma.backends.factory import get_backend


class BaseLoss:
    def __init__(self, per_image: bool = False, per_channel: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.per_image = per_image
        self.per_channel = per_channel

    def _get_reduction_axes(self, ndim: int) -> tuple:
        """Determina automaticamente quali assi sommare."""
        axes = list(self.be.spatial_axes(ndim))
        
        # Se NON vogliamo la loss per immagine, sommiamo anche l'asse del Batch (0)
        if not self.per_image:
            axes.append(0)
            
        # Se NON vogliamo la loss per canale, sommiamo anche l'asse del Canale
        if not self.per_channel:
            channel_axis = 1 if self.be.channel_format == "channels_first" else ndim - 1
            axes.append(channel_axis)
            
        return tuple(sorted(axes))

    def __call__(self, y_true, y_pred):
        pass