import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets


class Generator(nn.Module):
    def __init__(
        self, 
        input_dim=100,
        context_dim=10,
        num_classes=10,
        ):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, context_dim)
        self.mean = nn.Sequential(
            nn.Linear(input_dim + context_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(input_dim + context_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Softplus()
        )
        
    def forward(self, z, labels):
        embedding = self.label_embedding(labels)
        embedding = torch.cat([z, embedding], -1)
        mean = self.mean(embedding)
        std = self.std(embedding)
        
        eps = torch.randn_like(std)
        samples = mean + eps * std 
        return samples.view(z.size(0), 28, 28)
    
    def log_prob(self, x, z, labels):
        embedding = self.label_embedding(labels)
        embedding = torch.cat([z, embedding], -1)
        mean = self.mean(embedding)
        std = self.std(embedding).clip(1e-6, 1e6)
        std = torch.clamp(std / std.min(), 1, 100)
        
        # norm_dist = dists.Normal(mean, std)
        # log_probs = norm_dist.log_prob(x.view(x.size(0), 784))
        # log_prob = log_probs.sum(dim=-1)
        log_prob = -torch.sum((x.view(x.size(0), -1) - mean) ** 2 / std ** 2, dim=-1)
        return log_prob
    
    def sample(self, num_samples=1, labels=None):
        device = next(self.parameters()).device
        if labels is None:
            labels = torch.randint(0, 10, (num_samples,))
            labels = labels.to(device)
        z = torch.randn(num_samples, 100).to(device)
        return self.forward(z, labels)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out


def pretrain_gan(
    batch_size=64, 
    num_epochs=200, 
    learning_rate=1e-3, 
    device='cpu',
    beta1=0.5,
    beta2=0.999,
    ):
    
    cuda = False if device == 'cpu' else True
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Configure data loader
    os.makedirs("./data/mnist", exist_ok=True)
    dataloader = DataLoader(
        datasets.MNIST(
            "../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    adversarial_loss = torch.nn.MSELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    pbar = tqdm(total=num_epochs * len(dataloader))
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)
            labels = Variable(labels.type(torch.LongTensor)).to(device)

            # train generator
            optimizer_G.zero_grad()
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100)))).to(device)
            gen_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # train discriminator
            optimizer_D.zero_grad()
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            pbar.update(1)
            if i % 10 == 0:
                pbar.set_description(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
            
        # save models
        torch.save(generator.state_dict(), "./weights/generator_test.pth")
        torch.save(discriminator.state_dict(), "./weights/discriminator_test.pth")
            
    return generator, discriminator


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = pretrain_gan(device=device, num_epochs=20, learning_rate=0.0001)