from typing import Iterable, Optional
import torch


class BasePreferenceDataset(torch.utils.data.Dataset):
    """
    Base dataset class for preference alignment with flow matching.
    Sources should be list of less preferred samples, $\{y_{i}^{-}\}_{i=1}^{N}$.
    Targets is a list of preferred samples, $\{y_{i}^{+}\}_{i=1}^{N}$, with same length.
    If contexts $\{x_{i}\}_{i=1}^{N}$ are provided, 
    `__getitem__` outputs a pair $(x_{i}, y_{i}^{+}, y_{i}^{-})$.
    If contexts are not provided, outputs a pair $(y_{i}^{+}, y_{i}^{-})$.
    """
    def __init__(
            self, 
            sources: Iterable[torch.Tensor], 
            targets: Iterable[torch.Tensor], 
            contexts: Optional[Iterable[torch.Tensor]]=None
        ) -> None:
        self.contexts = contexts
        self.targets = targets
        self.sources = sources

    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, idx): 
        source = self.sources[idx].unsqueeze(0)
        target = self.targets[idx].unsqueeze(0)
        if self.contexts is None:
            return target, source
        
        context = self.contexts[idx]
        return context, target, source

