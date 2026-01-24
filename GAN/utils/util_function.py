import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple,Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
import random

def update_average(model_tgt, model_src, beta):
    """
    function to calculate the Exponential moving averages for the Generator weights
    This function updates the exponential average weights based on the current training
    Args:
        model_tgt: target model
        model_src: source model
        beta: value of decay beta
    Returns: None (updates the target model)
    """

    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

def adjust_dynamic_range(
    data: Tensor,
    drange_in: Optional[Tuple[float, float]] = (-1.0, 1.0),
    drange_out: Optional[Tuple[float, float]] = (0.0, 1.0),
):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0])
        )
        bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
        data = data * scale + bias

    return torch.clamp(data, min=drange_out[0], max=drange_out[1])


def post_process_generated_images(imgs: Tensor) -> np.array:
    imgs = adjust_dynamic_range(
        imgs.permute(0, 2, 3, 1), drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )
    return (imgs * 255.0).detach().cpu().numpy().astype(np.uint8)


def preprocess_img(dataset, size=128):
    transform = T.Compose([
        T.ToImage(),  # PIL/HF Image -> Tensor(tv_tensors.Image)
        T.CenterCrop(size),
        T.Resize((size, size)), # Ensure exact size output if crop is different or image is smaller
        T.ToDtype(torch.float32, scale=True),  # uint8 -> float32, [0,255]->[0,1]
        # (GAN이면 보통 아래 Normalize도 추가)
        # T.Normalize(mean=(0.5,)*3, std=(0.5,)*3),  # [0,1]->[-1,1]
    ])

    def transform_fn(ex):
        # Handle grayscale images by converting to RGB if needed
        # (CelebA is RGB but robust code helps)
        # However, T.ToImage() handles PIL.
        ex["pixel_values"] = transform(ex["image"])
        return ex

    for split in dataset.keys():
        dataset[split].set_transform(transform_fn)

    return dataset

def stargan_preprocess_img(dataset, opt, size=128):
    transform = T.Compose([
                            T.ToImage(),
                            T.RandomHorizontalFlip(),
                            T.CenterCrop(opt.crop_size),
                            T.Resize(opt.image_size),
                            T.ToDtype(torch.float32, scale=True),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ])

    def transform_fn(ex):
        # Handle grayscale images by converting to RGB if needed
        # (CelebA is RGB but robust code helps)
        # However, T.ToImage() handles PIL.
        ex["pixel_values"] = transform(ex["image"])

        return ex

    for split in dataset.keys():
        dataset[split].set_transform(transform_fn)

    return dataset

def stargan_collate_pixel_values(batch,opt):
    # batch: List[dict], 각 dict에 'pixel_values'가 있음
    labels = torch.tensor([[int(b[attr]) for attr in opt.selected_attrs] for b in batch],dtype=torch.float32)
    return torch.stack([b["pixel_values"] for b in batch], dim=0), labels

def stargan_get_data_loader(
    dataset: Dataset, batch_size: int, num_workers: int = 3
) -> DataLoader:
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=stargan_collate_pixel_values
    )


def collate_pixel_values(batch):
    # batch: List[dict], 각 dict에 'pixel_values'가 있음
    return torch.stack([b["pixel_values"] for b in batch], dim=0)

def get_data_loader(
    dataset: Dataset, batch_size: int, num_workers: int = 3
) -> DataLoader:
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_pixel_values
    )


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return, dim=0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)