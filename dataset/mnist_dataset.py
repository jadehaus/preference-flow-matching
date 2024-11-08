from typing import Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import Generator
from .base_dataset import BasePreferenceDataset


class RewardFunction():
    def __init__(self, classifier, device='cpu'):
        self.device = device
        self.classifier = classifier
        
    def to(self, device):
        self.device = device
        self.classifier.to(device)
        return self

    def pred_label(self, y):
        pad_y = F.pad(y, (2, 2, 2, 2), value=0)
        logits = self.classifier(pad_y)
        return torch.argmax(logits, dim=1)

    def __call__(self, y, x=None):
        if x is None:
            x = self.pred_label(y).to(self.device)
        pad_y = F.pad(y, (2, 2, 2, 2), value=0)
        logits = self.classifier(pad_y)
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs[torch.arange(len(probs)), x]
        return probs


def generate_image(
    generator: Generator, 
    sample_size: int=1, 
    condition: Optional[int]=None, 
    device: torch.device='cpu'
    ) -> Tuple[torch.Tensor, int]:
    """
    Generates a MNIST image sample with a provided conditional GAN.
    """
    generator.eval().to(device)
    if condition is None:
        condition = torch.randint(0, 10, (sample_size,))
        condition = condition.to(device)
    
    z = torch.randn(sample_size, 100).to(device)
    generated_image = generator(z, condition)
    return generated_image, condition


def generate_mnist_dataset(
    generator: Generator,
    reward_function: RewardFunction, 
    sample_size: int=10000, 
    device: torch.device='cpu', 
    verbose: bool=True,
    ) -> BasePreferenceDataset:
    """
    Generates a MNIST preference dataset, 
    using the provide generator and a reward function.
    """
    generator.to(device)
    reward_function.to(device)
    
    if verbose:
        pbar = tqdm(total=sample_size)
    
    contexts = []
    positive = []
    negative = []

    while len(contexts) < sample_size:
        with torch.no_grad():
            images, labels = generate_image(generator, sample_size=100, device=device)
            images = images.view(100, 1, 28, 28)
            rewards = reward_function(images, labels)
            
        for label in range(10):
            idxs = torch.where(labels == label)[0]
            if len(idxs) < 2:
                continue
            
            _, rewards_idx = torch.sort(rewards, descending=True)
            rewards_idx = rewards_idx[labels[rewards_idx] == label]
            
            y1 = images[rewards_idx[0]].squeeze(0)
            y0 = images[rewards_idx[-1]].squeeze(0)

            positive.append(y1)
            negative.append(y0)
            contexts.append(label)

            if verbose:
                pbar.update(1)
            if len(contexts) >= sample_size:
                return BasePreferenceDataset(positive, negative, contexts)
                
    return BasePreferenceDataset(positive, negative, contexts)

