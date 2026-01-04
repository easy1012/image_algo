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
n_classes = 10

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
    def __init__(self,
                 latent_dim,
                 img_shape,
                 n_classes):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        self.model = nn.Sequential(
            *self.block(latent_dim + n_classes, 128, normalize=False),
            *self.block(128 ,256),
            *self.block(256,512),
            *self.block(512,1024),
            nn.Linear(1024, img_shape[0] * img_shape[1] * img_shape[2]),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        input_z = torch.cat((self.label_emb(labels).squeeze(1), noise), -1)
        img = self.model(input_z)
        img = img.view(-1, *self.img_shape)
        return img
    
    def block(self, input, output, normalize = True):
        layers = [nn.Linear(input, output)]
        
        if normalize:
            layers.append(nn.BatchNorm1d(output, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace= True))
        return layers
        
    
class Discriminator(nn.Module):
    def __init__(self, img_shape, n_classes):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        
        self.label_embedding = nn.Embedding(n_classes,n_classes)
        self.model = nn.Sequential(
            nn.Linear(n_classes + img_shape[0] *img_shape[1] * img_shape[2], 512),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Linear(512, 1)
        )
        
    def forward(self, img, labels):
        input_x = torch.cat((img.view(img.shape[0],-1),
                            self.label_embedding(labels).squeeze(1)), -1)
        pred = self.model(input_x)
        return pred
    
    
    
G = Generator(latent_dim, image_shape, 10)
D = Discriminator(image_shape, 10)

# loss = torch.nn.BCELoss()
loss = torch.nn.MSELoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr = lr, betas = (b1,b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr = lr, betas = (b1,b2))


if __name__ in "__main__":    
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            
            real = torch.full((imgs.size(0),1), 
                            1.0,
                            requires_grad= False)
            fake = torch.full((imgs.size(0),1),
                            0.0,
                            requires_grad= False)
            
            real_imgs = imgs
            
            optimizer_G.zero_grad()
            z = torch.normal(mean = 0.0,
                            std = 1,
                            size = (imgs.shape[0],latent_dim))
            
            gen_labels = torch.randint(0, 10, (imgs.shape[0], 1))
            gen_imgs = G(z, gen_labels)
            fake_pred = D(gen_imgs, gen_labels)
            g_loss = loss(fake_pred, real)
            
            g_loss.backward()
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            real_pred = D(real_imgs, labels)
            d_real_loss = loss(real_pred, real)
            
            fake_pred = D(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
        print(f"epoch : {epoch} g_loss : {g_loss} d_loss : {d_loss}")