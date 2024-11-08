import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=128, output_dim=None, 
        num_layers=3, time_varying=True, context_dim=None
        ) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.time_dim = 1 if time_varying else 0
        self.context_dim = 0 if context_dim is None else context_dim
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_layer = nn.Sequential(
            nn.Linear(
                input_dim + self.time_dim + self.context_dim, 
                hidden_dim
            ),
            nn.ReLU(),
        )
            
        self.hidden_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, x, context=None):
        x = x.reshape((-1, x.shape[-1]))
        if context is not None:
            emb = torch.cat([x, t[:, None], context], dim=-1)
        else:
            emb = torch.cat([x, t[:, None]], dim=-1)
        emb = self.input_layer(emb)
        return self.hidden_layers(emb)
    
