import pickle
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.nn import functional as F
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import logging
import os
from matplotlib import pyplot
from PIL import Image

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class AE(LightningModule):

    def __init__(
        self,
        lr: float = 1e-3,
        kl_coeff: float = 0.01,
        latent_dim: int = 768,
        enc_out_dim: int = 768,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(True),
            nn.Linear(768, 768),
        )

        self.decoder = nn.Sequential(          
            nn.Linear(768, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 5488),
        )
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z
    
    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)
        recon_loss = F.mse_loss(x_hat, y)
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = recon_loss + kl 

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class VAE(LightningModule):

    def __init__(
        self,
        input_height: int,
        enc_type: str = "default",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 5488,
        kl_coeff: float = 0.01,
        latent_dim: int = 5488,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        if enc_type == "default":
            if input_height == 224:
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 224, (3,3), stride=(1,1), padding=(1,1)),  # 224 x 224 x 224
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 224 x 112 x 112
        
                    nn.Conv2d(224, 112, (3,3), stride=(1,1), padding=(1,1)),  # 112 x 112 x 112
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 112 x 56 x 56
        
                    nn.Conv2d(112, 56, (3,3), stride=(1,1), padding=(1,1)),  # 56 x 56 x 56
                    nn.ReLU(True),
                    nn.MaxPool2d(2),  # 56 x 28 x 28
        
                    nn.Conv2d(56, 7, (3, 3), stride=(1,1), padding=(1,1)),  # 7 x 28 x 28
                    nn.Flatten()  # 5488 x 1 x 1
                )
                 
                self.decoder = nn.Sequential(    
                    View([-1, 7, 28, 28]),
                    nn.ConvTranspose2d(7, 56, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 32 x 64 x 64
                
                    nn.ConvTranspose2d(56, 112, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(112, 224, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2),  # 64 x 128 x 128
        
                    nn.ConvTranspose2d(224, 1, (3,3), stride=(1,1), padding=(1,1)),
                    nn.Sigmoid()
                )
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q
    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)
        ls = nn.BCELoss()
        recon_loss = ls(x_hat, x)
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = recon_loss + kl 

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)