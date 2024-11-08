import torch
from torchcfm.models.unet import UNetModel


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class UNet(torch.nn.Module):
    def __init__(
        self, 
        dim: tuple=(1, 32, 32), 
        class_cond: bool=False, 
        num_classes: int=None
        )-> None:
        super().__init__()
        self.model = UNetModel(
            dim=dim, 
            num_channels=32,
            num_res_blocks=1,
            class_cond=class_cond,
            num_classes=num_classes,
        )
               
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self
        
    def forward(self, t, y, x):
        v = self.model(t, y, x)
        return v
