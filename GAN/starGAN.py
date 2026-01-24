import torch
from torch.utils.data import Dataset, DataLoader
from utils.stargan_DG import Generator, Discriminator
from datasets import load_dataset
from torchvision.transforms import v2 as T

import torch.nn.functional as F

from utils.util_function import stargan_preprocess_img, stargan_collate_pixel_values,stargan_get_data_loader


class StarGAN():
    def __init__(self, opt):
        self.G = Generator(opt.g_conv_dim, opt.c_dim, opt.g_repeat_num)
        self.D = Discriminator(opt.image_size, opt.d_conv_dim, opt.c_dim, opt.d_repeat_num)
        self.opt = opt
        self.train_dataset = self._make_dataset()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.opt.g_lr, [self.opt.beta1, self.opt.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.opt.d_lr, [self.opt.beta1, self.opt.beta2])

    def _make_dataset(self):
        dataset = load_dataset('flwrlabs/celeba')
        return stargan_preprocess_img(dataset, self.opt)['train']

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.opt.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def train(self,):
        train_dataloader = stargan_get_data_loader(self.train_dataset,
                                                   self.opt.batch_size)
        
        start_iters = 0
        init_dataloader = iter(train_dataloader)
        
        for i in range(start_iters, self.opt.num_iters):
            x_real, label_org = next(init_dataloader)
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.opt.device)           # Input images.
            c_org = c_org.to(self.opt.device)             # Original domain labels.
            c_trg = c_trg.to(self.opt.device)             # Target domain labels.
            label_org = label_org.to(self.opt.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.opt.device)     # Labels for computing classification loss.

            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, label_org, size_average=False) / out_cls.size(0)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.opt.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_loss_real + d_loss_fake + self.opt.lambda_cls * d_loss_cls + self.opt.lambda_gp * d_loss_gp

            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            d_loss.backward()
            self.d_optimizer.step()

            loss = {}
            loss['D_loss_real'] = d_loss_real.item()
            loss['D_loss_fake'] = d_loss_fake.item()
            loss['D_loss_cls'] = d_loss_cls.item()
            loss['D_loss_gp'] = d_loss_gp.item()


            if (i + 1) % self.opt.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, label_trg, size_average=False) / out_cls.size(0)
                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.opt.lambda_rec * g_loss_rec + self.opt.lambda_cls * g_loss_cls

                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                
    def predict(self):
        pass
    