import torch
from catopuma.core.losses.segmentation import DiceLoss as AgnosticDiceLoss

class DiceLoss(torch.nn.Module):
    def __init__(self, per_image: bool = False, per_channel: bool = False, smooth: float = 1e-6):
        super().__init__()
        # Inizializziamo la loss agnostica forzando il backend 'torch'
        self.agnostic_dice = AgnosticDiceLoss(
            backend_name="torch", 
            per_image=per_image, 
            per_channel=per_channel,
            smooth=smooth
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Interfaccia standard PyTorch.
        y_pred: (B, C, H, W) o (B, C, D, H, W)
        y_true: stessa shape di y_pred (one-hot o mask)
        """
        # Chiamiamo la logica agnostica. 
        # Essendo il backend 'torch', l'operazione rimane nel grafo di autograd.
        return self.agnostic_dice(y_true, y_pred)