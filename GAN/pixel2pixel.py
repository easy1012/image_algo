import os
import numpy as np
import math
from datasets import load_dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from PIL import Image
import io

import matplotlib.pyplot as plt
# from utils.util_function import preprocess_img


class pixel2pixel:
    def __init__(self, gen, dis, device, lr, b1, b2, lambda_pixel):
        self.device = device
        self.generator = gen.to(device)
        self.discriminator = dis.to(device)
        
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()
        self.dataset = self._make_dataset()
        self.lambda_pixel = lambda_pixel

    def _make_dataset(self):
        data = load_dataset('huggan/facades')
        dataset = self.preprocess_img(data)
        return dataset['train']

    def preprocess_img(self,dataset, size=256):
        transform_ = transforms.Compose(
            [transforms.Resize((size, size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        def transform_fn(ex):
            # Handle grayscale images by converting to RGB if needed
            # (CelebA is RGB but robust code helps)
            # However, T.ToImage() handles PIL.

            ex_a = list(map(lambda x : Image.open(io.BytesIO(x['bytes'])).convert('RGB'), ex["imageA"]))
            ex_b = list(map(lambda x : Image.open(io.BytesIO(x['bytes'])).convert('RGB'), ex["imageA"]))
            ex["pixel_image_A"] = [transform_(i) for i in ex_a]
            ex["pixel_image_B"] = [transform_(i) for i in ex_b]
            return ex

        for split in dataset.keys():
            dataset[split].set_transform(transform_fn)

        return dataset

    def collate_pixel_values(self,batch):
        # batch: List[dict], 각 dict에 'pixel_values'가 있음
        return {'a' : torch.stack([b["pixel_image_A"] for b in batch], dim=0), 'b' : torch.stack([b["pixel_image_B"] for b in batch], dim=0)}

    @staticmethod
    def get_data_loader(
        dataset: Dataset, batch_size: int, num_workers: int = 3
    ) -> DataLoader:
    
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=self.collate_pixel_values
        )

    def train(self, 
            batch_size,
            num_epochs,
            dataloader,
            lr,
            b1,
            b2 ):

        dataloader = self.get_data_loader(self.dataset, batch_size)
        optimizer_G = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))

        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                real_a = batch['a'].to(self.device)
                real_b = batch['b'].to(self.device)
                
                valid = torch.ones(real_a.size(0), 1).to(self.device)
                fake = torch.zeros(real_a.size(0), 1).to(self.device)
                
                optimizer_G.zero_grad()

                fake_b = self.generator(real_a)
                pred_fake = self.discriminator(fake_b, real_a)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                loss_pixel = self.criterion_pixelwise(fake_b, real_b)

                loss_G = loss_GAN + self.lambda_pixel * loss_pixel
                loss_G.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                
                pred_real = self.discriminator(real_b, real_a)
                loss_real = self.criterion_GAN(pred_real, valid)
                
                pred_fake = self.discriminator(fake_b.detach(), real_a)
                loss_fake = self.criterion_GAN(pred_fake, fake)
                
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optimizer_D.step()

                batches_done = epoch * len(dataloader) + i