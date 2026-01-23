import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        self.files_A = sorted(glob.glob(os.path.join(root, 'trainA') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'trainB') + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
)

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB")
)
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")
)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class CycleGAN():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG_A2B = Generator(self.opt.input_nc, self.opt.output_nc).to(self.device)
        self.netG_B2A = Generator(self.opt.output_nc, self.opt.input_nc).to(self.device)
        self.netD_A   = Discriminator(self.opt.input_nc).to(self.device)
        self.netD_B   = Discriminator(self.opt.output_nc).to(self.device)

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=self.opt.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        self.transforms_ = [
            transforms.Resize(int(self.opt.size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.opt.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

    def train(self):
        dataloader = DataLoader(
            ImageDataset(self.opt.file_root, transforms_=self.transforms_, unaligned=True),
            batch_size=self.opt.batch_size, shuffle=True, num_workers=3,drop_last=True
        )
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, batch in enumerate(dataloader):
                # Set model input (Variable/input_A/input_B 제거)
                real_A = batch["A"].to(device)
                real_B = batch["B"].to(device)

                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                same_B = self.netG_A2B(real_B)
                loss_identity_B = self.criterion_identity(same_B, real_B) * 5.0

                same_A = self.netG_B2A(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0

                # GAN loss (target_real을 ones_like로 즉석 생성)
                fake_B = self.netG_A2B(real_A)
                pred_fake_B = self.netD_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

                fake_A = self.netG_B2A(real_B)
                pred_fake_A = self.netD_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * 10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * 10.0

                # Total loss
                loss_G = (
                    loss_identity_A
                    + loss_identity_B
                    + loss_GAN_A2B
                    + loss_GAN_B2A
                    + loss_cycle_ABA
                    + loss_cycle_BAB
                )
                loss_G.backward()
                self.optimizer_G.step()

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                pred_real_A = self.netD_A(real_A)
                loss_D_real_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

                # Fake loss
                fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake_A = self.netD_A(fake_A_.detach())
                loss_D_fake_A = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

                # Total loss
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
                loss_D_A.backward()
                self.optimizer_D_A.step()

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real_B = self.netD_B(real_B)
                loss_D_real_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

                # Fake loss
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake_B = self.netD_B(fake_B_.detach())
                loss_D_fake_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

                # Total loss
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
                loss_D_B.backward()
                self.optimizer_D_B.step()

                # (선택) 로깅 출력
                # if i % 50 == 0:
                #     print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                #           f"[D loss: {(loss_D_A + loss_D_B).item():.4f}] [G loss: {loss_G.item():.4f}]")

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

    def test(self):
        pass



    def __str__(self):
        return "CycleGAN"