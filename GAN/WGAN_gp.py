import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd


latent_dim = 100
image_shape = (1, 28, 28)
lr = 0.0001
b1,b2 = 0.5, 0.999
epochs = 100
n_critic = 5
lambda_gp = 10

dataloader = DataLoader(datasets.MNIST(
    "../dataset/mnist",
    train = True,
    download= True,
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])])),
    batch_size = 64,
    shuffle=True
    )


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *self.block(latent_dim ,128, normalize=False),
            *self.block(128 ,256),
            *self.block(256,512),
            *self.block(512,1024),
            nn.Linear(1024, img_shape[0] * img_shape[1] *img_shape[2]),
            nn.Tanh()
        )
    
    def forward(self,z):
        img = self.model(z)
        img = img.view(-1, *self.img_shape)
        return img
    
    def block(self,input, output, normalize = True):
        layers = [nn.Linear(input, output)]
        
        if normalize:
            layers.append(nn.BatchNorm1d(output, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace= True))
        
        return layers
    
    
class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape[0] *img_shape[1] * img_shape[2], 512),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(256, 1)
        )
        
    def forward(self, img):
        flat_img = img.view(img.size(0), -1)
        pred = self.model(flat_img)
        return pred
    
    
G = Generator(latent_dim, image_shape)
D = Discriminator(image_shape)

loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr = lr, betas = (b1,b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr = lr, betas = (b1,b2))

def gradient_penalty(D, batch_size, real_img, fake_img):
    alpha = torch.randn(batch_size,1, 28, 28)
    interpolates = (alpha * real_img + ((1 - alpha) * fake_img)).requires_grad_(True)
    
    d_interpolates = D(interpolates)
    
    fake = torch.full((imgs.size(0),1),
                          1.0,
                          requires_grad= False)
    
    gradients = autograd.grad(outputs = d_interpolates,
                              inputs = interpolates,
                              grad_outputs = fake,
                              create_graph = True,
                              retain_graph = True,
                              only_inputs = True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    GP = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
    
    return GP

if __name__ == '__main__':
    current_iters = 0
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs= imgs
            
            optimizer_D.zero_grad()        
            z = torch.normal(mean = 0.0,
                        std = 1,
                        size = (imgs.shape[0],latent_dim))
            
            fake_imgs = G(z)
            
            real_pred = D(real_imgs)
            fake_pred = D(fake_imgs)
            GP = gradient_penalty(D,imgs.size()[0], real_imgs, fake_imgs)
            
            d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) +lambda_gp * GP
            
                    
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            
            if i % n_critic == 0:
                fake_imgs = G(z)
                fake_pred = D(fake_imgs)
                g_loss = -torch.mean(fake_pred)
                g_loss.backward()
                optimizer_G.step()
                current_iters += n_critic
            
        print(f"epoch : {epoch} g_loss : {g_loss} d_loss : {d_loss}")