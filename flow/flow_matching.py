import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchdyn
import torchdiffeq
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *


class NodeWrapper(nn.Module):
    def __init__(
            self, 
            model, 
            context=None,
        ) -> None:
        super().__init__()
        self.model = model
        if context is not None:
            context = context[None, :]
        self.context = context

    def forward(self, t, x, *args, **kwargs):
        if len(x.shape) == 1:
            x = x[None, :]
        t = t.repeat(x.shape[0])
        out = self.model(t, x, self.context)
        return out.squeeze(0)


class OptimalTransportConditionalFlowMatching():
    """
    PyTorch implementation of the Optimal Transport Conditional Flow Matching algorithm.
    Learns a ODE flow from source to target trajectories, given context.
    Wrapper class for the torchcfm library.
    (Tong, Alexander, et al. 2023)

    The input model should be a neural network that takes input (t, y, x) and outputs v:
        t: time
        y: target trajectory
        x: context
        v: vector field
    
    In case of unconditional flow matching, the context is not used,
    and the model should take input (t, y) and output v.
    For such a model, set conditional=False in the fit method.
    """
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(self.device)

    def __call__(self, source, context=None, **kwargs):
        return self.compute_target(source, context=context, **kwargs)
    
    def fit(
            self, 
            dataset: torch.utils.data.Dataset, 
            num_epochs: int=100, 
            batch_size: int=64, 
            learning_rate: float=1e-3, 
            sigma: float=0.1,
            conditional: bool=False,
            loss_scale=1,
            collate_fn=None,
            save_path=None,
            save_losses_path=None,
            save_interval=10,
        ):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=True, collate_fn=collate_fn
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        flow = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

        # training loop
        info = {
            'loss': [],
            '||u_t||_2': [],
            '||v_t||_2': [],
            '||u_t||_inf': [],
            '||v_t||_inf': [],
        }
        
        self.model.train()
        pbar = tqdm(
            range(num_epochs * len(dataloader)), 
            desc=f'Epoch 0 / {num_epochs}, Loss: ---'
        )
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                
                optimizer.zero_grad()

                if conditional:
                    x, y1, y0 = batch
                    y1 = y1.to(self.device)
                    y0 = y0.to(self.device)

                    t, yt, ut, _, x1 = flow.guided_sample_location_and_conditional_flow(y0.float(), y1.float(), y1=x)
                    x1 = x1.to(self.device)

                else:
                    y1, y0 = batch
                    y1 = y1.to(self.device)
                    y0 = y0.to(self.device)

                    t, yt, ut = flow.sample_location_and_conditional_flow(y0, y1)
                    x1 = None

                t = t.to(self.device)
                yt = yt.to(self.device)
                ut = ut.to(self.device)

                vt = self.model(t, yt, x1)
            
                loss = torch.mean((vt - ut) ** 2 * loss_scale)
                loss.backward()
                optimizer.step()
                
                vt_l_2 = torch.norm(vt, p=2, dim=1).mean().item()
                ut_l_2 = torch.norm(ut, p=2, dim=1).mean().item()
                vt_inf = torch.norm(vt, p=float('inf'), dim=1).mean().item()
                ut_inf = torch.norm(ut, p=float('inf'), dim=1).mean().item()
                
                info['loss'].append(loss.item())
                info['||u_t||_2'].append(ut_l_2)
                info['||v_t||_2'].append(vt_l_2)
                info['||u_t||_inf'].append(ut_inf)
                info['||v_t||_inf'].append(vt_inf)
                
                pbar.update(1)
                pbar.set_description(
                    f'Epoch {epoch + 1} / {num_epochs},  Loss: {loss.item():.5f}'
                )

                if i % save_interval == 0:
                    if save_losses_path is not None:
                        if not os.path.exists(save_losses_path):
                            os.makedirs(save_losses_path)
                        np.save(f"{save_losses_path}/loss.npy", np.array(info['loss']))

            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model, f"{save_path}/ckpt_epoch_{epoch}.pth")

        self.model.eval()
        return self.model, info

    def _compute_trajectory(
            self, 
            source, 
            context=None, 
            timesteps=10,
            use_torchdiffeq=True,
        ):
        source = source.to(self.device)
        if context is not None:
            context = context.to(self.device)

        self.model.eval()
        if use_torchdiffeq:
            with torch.no_grad():
                model = lambda t, y: self.model(t, y, context)
                traj = torchdiffeq.odeint(
                    model,
                    source,
                    torch.linspace(0, 1, timesteps).to(self.device),
                    atol=1e-4, 
                    rtol=1e-4,
                    method='dopri5',
                )
        else:
            with torch.no_grad():
                node = NeuralODE(
                    NodeWrapper(self.model, context=context),
                    solver='dopri5', sensitivity='adjoint',
                    atol=1e-4, rtol=1e-4
                )
                traj = node.trajectory(
                    source, 
                    t_span=torch.linspace(0, 1, timesteps, device=self.device),
                )
        return traj
    
    def compute_target(
            self, 
            source, 
            context=None,
            timesteps=10,
            use_torchdiffeq=True,
        ):
        traj = self._compute_trajectory(
            source, 
            context=context,
            timesteps=timesteps,
            use_torchdiffeq=use_torchdiffeq,
        )
        return traj[-1]
    
    def plot_trajectories(
            self, 
            source, 
            context=None,
            timesteps=10,
            use_torchdiffeq=True,
        ):
        traj = self._compute_trajectory(
            source, 
            context=context,
            timesteps=timesteps,
            use_torchdiffeq=use_torchdiffeq,
        ).cpu().numpy()

        n = 2000
        plt.figure(figsize=(6, 6))
        plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
        plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
        plt.legend(["Source p0", "Flow", "Target p1"])
        plt.xticks([])
        plt.yticks([])
        plt.show()