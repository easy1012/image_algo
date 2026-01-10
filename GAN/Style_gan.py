
import math
import random
import os
import copy
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.stylegan_DG import Generator, Discriminator
from utils.util_function import adjust_dynamic_range, get_data_loader, preprocess_img, update_average
from datasets import load_dataset

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths # return path_lengths for logging if needed

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]

class TrainableGenerator(nn.Module):
    """
    Wrapper for Generator to handle list inputs cleanly across DataParallel if needed,
    although standard DataParallel splits tensors.
    Here we accept explicit z1, z2 arguments to ensuring splitting works.
    """
    def __init__(self, generator):
        super().__init__()
        self.gen = generator

    def forward(self, z1, z2=None, inject_index=None, return_latents=False):
        styles = [z1]
        if z2 is not None:
            styles.append(z2)
        return self.gen(styles, return_latents=return_latents, inject_index=inject_index)

class StyleGAN:
    def __init__(
        self,
        gen: Generator,
        dis: Discriminator,
        device=torch.device("cpu"),
        use_ema: bool = True,
        ema_beta: float = 0.999,
        r1_gamma: float = 10.0,
        g_reg_every: int = 4,
        d_reg_every: int = 16,
        mixing_prob: float = 0.9,
    ):
        self.device = device
        self.size = gen.size
        self.style_dim = gen.style_dim
        
        # Prepare Generator Wrapper
        # If running on GPU, we move modules to device first
        gen.to(device)
        self.gen_module = TrainableGenerator(gen).to(device)
        
        self.dis = dis.to(device)
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        
        # Training Hyperparameters
        self.r1 = r1_gamma
        self.g_reg_every = g_reg_every
        self.d_reg_every = d_reg_every
        self.mixing_prob = mixing_prob
        
        # Path regularization state
        self.mean_path_length = 0
        
        self.dataset = self._make_dataset() # Initialize dataset

        if self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda"):
            self.gen = DataParallel(self.gen_module)
            self.dis = DataParallel(self.dis)
        else:
            self.gen = self.gen_module

        # EMA Setup
        if self.use_ema:
            self.gen_shadow = copy.deepcopy(gen) # Deep copy the core Generator
            self.gen_shadow.eval()
            update_average(self.gen_shadow, gen, beta=0) # Init

    def _make_dataset(self):
        dataset = load_dataset('flwrlabs/celeba')
        return preprocess_img(dataset, size=self.size)['train']
    
    def _check_grad_ok(self, network: nn.Module) -> bool:
        grad_ok = True
        for _, param in network.named_parameters():
            if param.grad is not None:
                param_ok = (
                    torch.sum(torch.isnan(param.grad)) == 0
                    and torch.sum(torch.isinf(param.grad)) == 0
                )
                if not param_ok:
                    grad_ok = False
                    break
        return grad_ok

    def get_save_info(self, gen_optim, dis_optim):
        # Unwrap DataParallel if present for saving
        gen_state = self.gen.module.gen.state_dict() if isinstance(self.gen, DataParallel) else self.gen_module.gen.state_dict()
        dis_state = self.dis.module.state_dict() if isinstance(self.dis, DataParallel) else self.dis.state_dict()
        
        save_info = {
            "generator": gen_state,
            "discriminator": dis_state,
            "gen_optim": gen_optim.state_dict(),
            "dis_optim": dis_optim.state_dict(),
        }
        
        if self.use_ema:
             save_info["shadow_generator"] = self.gen_shadow.state_dict()

        return save_info

    def train(
        self,
        epochs: int,
        batch_size: int,
        gen_lr: float = 0.002,
        dis_lr: float = 0.002,
        num_workers: int = 3,
        save_dir=Path("./train_stylegan"),
        checkpoint_factor: int = 10,
        feedback_factor: int = 50, # Print every N steps
    ):
        print(f"Loaded dataset with {len(self.dataset)} images.")
        
        data_loader = get_data_loader(self.dataset, batch_size, num_workers)
        
        # Adjust learning rates for lazy regularization
        g_optim = Adam(self.gen.parameters(), lr=gen_lr * self.g_reg_every / (self.g_reg_every + 1), betas=(0.0, 0.99))
        d_optim = Adam(self.dis.parameters(), lr=dis_lr * self.d_reg_every / (self.d_reg_every + 1), betas=(0.0, 0.99))

        # Setup Logging
        save_dir = Path(save_dir)
        model_dir = save_dir / "models"
        log_dir = save_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir / "tensorboard"))

        # Fixed noise for comparison
        fixed_input = torch.randn(16, self.style_dim, device=self.device)
        
        self.gen.train()
        self.dis.train()

        global_step = 0
        
        print("Starting training...")
        
        for epoch in range(1, epochs + 1):
            for i, batch in enumerate(data_loader):
                global_step += 1
                
                real_img = batch.to(self.device)
                current_batch_size = real_img.shape[0]

                # =================================================================================== #
                #                             1. Train Discriminator                                  #
                # =================================================================================== #
                
                noise = mixing_noise(current_batch_size, self.style_dim, self.mixing_prob, self.device)
                
                # Unwrap noise for forward call (z1, z2 args)
                if len(noise) == 1:
                    fake_img, _ = self.gen(noise[0], return_latents=False)
                else:
                    fake_img, _ = self.gen(noise[0], noise[1], return_latents=False)
                
                real_pred = self.dis(real_img)
                fake_pred = self.dis(fake_img.detach())
                
                d_loss = d_logistic_loss(real_pred, fake_pred)
                
                self.dis.zero_grad()
                d_loss.backward()
                d_optim.step()
                
                d_regularize = global_step % self.d_reg_every == 0
                if d_regularize:
                    real_img.requires_grad = True
                    real_pred = self.dis(real_img)
                    r1_loss = d_r1_loss(real_pred, real_img)
                    
                    self.dis.zero_grad()
                    (self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]).backward() # 0*.. to keep graph
                    d_optim.step()
                    real_img.requires_grad = False
                    
                    writer.add_scalar("Loss/R1", r1_loss.item(), global_step)

                # =================================================================================== #
                #                               2. Train Generator                                    #
                # =================================================================================== #
                
                noise = mixing_noise(current_batch_size, self.style_dim, self.mixing_prob, self.device)
                
                if len(noise) == 1:
                    fake_img, _ = self.gen(noise[0], return_latents=False)
                else:
                    fake_img, _ = self.gen(noise[0], noise[1], return_latents=False)

                fake_pred = self.dis(fake_img)
                g_loss = g_nonsaturating_loss(fake_pred)
                
                self.gen.zero_grad()
                g_loss.backward()
                g_optim.step()
                
                g_regularize = global_step % self.g_reg_every == 0
                if g_regularize:
                    path_batch_size = max(1, current_batch_size // self.g_reg_every)
                    noise = mixing_noise(path_batch_size, self.style_dim, self.mixing_prob, self.device)
                    
                    # PPL needs latents (w)
                    if len(noise) == 1:
                        fake_img, latents = self.gen(noise[0], return_latents=True)
                    else:
                        fake_img, latents = self.gen(noise[0], noise[1], return_latents=True)
                    
                    path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                        fake_img, latents, self.mean_path_length
                    )
                    
                    self.gen.zero_grad()
                    weighted_path_loss = 2 * self.g_reg_every * path_loss
                    if self.g_reg_every > 1:
                         weighted_path_loss += 0 * fake_img[0,0,0,0]
                    weighted_path_loss.backward()
                    g_optim.step()
                    
                    writer.add_scalar("Loss/PathLength", path_lengths.mean().item(), global_step)

                # EMA Update
                if self.use_ema:
                    # Unwrap current generator module
                    current_gen = self.gen.module.gen if isinstance(self.gen, DataParallel) else self.gen_module.gen
                    update_average(self.gen_shadow, current_gen, self.ema_beta)

                # Logging
                if i % feedback_factor == 0:
                    print(f"Epoch {epoch} [{i}/{len(data_loader)}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")
                    writer.add_scalar("Loss/G", g_loss.item(), global_step)
                    writer.add_scalar("Loss/D", d_loss.item(), global_step)
            
            # End of Epoch
            # Save Sample
            with torch.no_grad():
                # For sampling, use the shadow generator if EMA is used
                sample_gen = self.gen_shadow if self.use_ema else (self.gen.module.gen if isinstance(self.gen, DataParallel) else self.gen.gen)
                
                # Check signature of raw generator
                # It expects styles=[z]
                sample_noise = [fixed_input]
                sample_img, _ = sample_gen(sample_noise)
                
                sample_img = adjust_dynamic_range(sample_img, (-1,1), (0,1))
                save_image(
                    sample_img,
                    str(log_dir / f"sample_{epoch}.png"),
                    nrow=int(math.sqrt(len(fixed_input))),
                    normalize=False,
                )
            
            # Save Model
            if epoch % checkpoint_factor == 0 or epoch == epochs:
                torch.save(
                    self.get_save_info(g_optim, d_optim),
                    str(model_dir / f"checkpoint_{epoch}.pt")
                )

        print("Training complete.")
