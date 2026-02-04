from datasets import load_dataset
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.losses import SRGAN_GeneratorLoss
from utils.srGAN_DG import Generator, Discriminator
from PIL import Image
from torchvision.transforms import v2



class SRGAN():
    def __init__(self, opt):
        self.netG = Generator(opt.scale_factor)
        self.netD = Discriminator()
        self.generator_criterion = SRGAN_GeneratorLoss()

        self.optimizerG = optim.Adam(self.netG.parameters())
        self.optimizerD = optim.Adam(self.netD.parameters())

        self.data = self._make_dataset(opt.hr_size, opt.upscale)


    def _make_dataset(self, hr_size, upscale):
        dataset = load_dataset('flwrlabs/celeba')
        train_data = dataset['train']

        hr_t = train_hr_transform(hr_size)
        lr_t = train_lr_transform(hr_size, upscale)

        train_data = dataset["train"].map(
            preprocess,
            remove_columns=dataset["train"].column_names
        )
        return train_data

        
    @staticmethod
    def train_hr_transform(crop_size):
        return v2.Compose([
                            v2.Resize(crop_size + 32, interpolation=Image.BICUBIC),  # optional but recommended
                            v2.CenterCrop(crop_size),
        ])

    @staticmethod
    def train_lr_transform(crop_size, upscale_factor):
        return v2.Compose([
                            v2.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
                            ])

    @staticmethod
    def preprocess(example):
        img = example["image"]  
        hr = hr_t(img)           
        lr = lr_t(hr)            
        return {"hr": hr, "lr": lr}

    def train(self):
        train_loader = DataLoader(
                                self.data,                 
                                batch_size=16,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=sr_collate_fn
                                    )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        
            self.netG.train()
            self.netD.train()
            for data, target in train_bar:
                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                ############################
                # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                real_img = target
                # if torch.cuda.is_available():
                #     real_img = real_img.float().cuda()
                z = data
                # if torch.cuda.is_available():
                    # z = z.float().cuda()
                fake_img = self.netG(z)
                fake_out = self.netD(fake_img).mean()

                self.optimizerG.zero_grad()
                g_loss = self.generator_criterion(fake_out, fake_img, real_img)
                g_loss.backward()
                self.optimizerG.step()

                ############################
                # (2) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_out = self.netD(real_img).mean()
                fake_out = self.netD(fake_img.detach()).mean()
                d_loss = 1 - real_out + fake_out

                self.optimizerD.zero_grad()
                d_loss.backward()
                
                fake_img = self.netG(z)
                fake_out = self.netD(fake_img).mean()

                self.optimizerD.step()

                # loss for current batch before optimization 
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.item() * batch_size
                running_results['g_score'] += fake_out.item() * batch_size
        
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
    