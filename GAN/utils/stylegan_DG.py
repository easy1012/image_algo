import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from .custom_layers import EqualizedLinear, EqualizedConv2d, MinibatchStdDev

# Alias for compatibility with user code
EqualLinear = EqualizedLinear

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if upsample:
             layers.append(Upsample(blur_kernel))

        layers.append(
            EqualizedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=kernel_size // 2,
                bias=bias,
            )
        )

        if downsample:
            layers.append(Downsample(blur_kernel))

        if activate:
            layers.append(nn.LeakyReLU(0.2))

        super().__init__(*layers)

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1, pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1, pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.demodulate = demodulate

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, output_height, output_width = out.shape
            out = out.view(batch, self.out_channel, output_height, output_width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            input = input.view(1, batch * in_channel, input.shape[2], input.shape[3])
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, output_height, output_width = out.shape
            out = out.view(batch, self.out_channel, output_height, output_width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, output_height, output_width = out.shape
            out = out.view(batch, self.out_channel, output_height, output_width)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None]
        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.upsample_factor = upsample_factor

    def forward(self, input):
        channel = input.shape[1]
        kernel = self.kernel.expand(channel, -1, -1, -1)
        # print(f"DEBUG: Blur input {input.shape}, pad {self.pad}")
        input = F.pad(input, self.pad)
        
        return F.conv2d(
            input, 
            kernel, 
            groups=channel
        )

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel * (factor ** 2)
        self.register_buffer('kernel', kernel[None, None])

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = F.interpolate(input, scale_factor=self.factor, mode='bilinear', align_corners=False)
        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None])

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = F.avg_pool2d(input, 2)
        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class StyleConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)
        else:
            self.upsample = None

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = EqualizedConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channel, out_channel, 3, padding=1)
        self.skip = EqualizedConv2d(in_channel, out_channel, 1, bias=False)

        self.downsample = nn.AvgPool2d(2)

    def forward(self, input):
        out = F.leaky_relu(self.conv1(input), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = self.downsample(out)

        skip = self.downsample(input)
        skip = self.skip(skip)

        return (out + skip) / math.sqrt(2)

class Generator(nn.Module):
    def __init__(self, size,
    style_dim,
    n_mlp,
    channel_multiplier=2,
    blur_kernel=[1, 3, 3, 1],
    lr_mlp=0.01):

       super().__init__()
       self.size = size
       self.style_dim = style_dim
       self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

       self.input = ConstantInput(self.channels[4])
       self.conv1 = StyleConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
       self.to_rgb1 = ToRGB(self.channels[4], self.style_dim, upsample=False)
       self.log_size = int(math.log(self.size, 2))
       self.num_layers = (self.log_size - 2) * 2 + 1
       self.convs = nn.ModuleList()
       self.upsamples = nn.ModuleList()
       self.to_rgbs = nn.ModuleList()
       self.noise = nn.ModuleList()

       # Mapping Network
       layers = [EqualLinear(style_dim, style_dim, bias=True)] # bias=True inferred
       for i in range(1, n_mlp):
            layers.append(nn.LeakyReLU(0.2))
            layers.append(EqualLinear(style_dim, style_dim, bias=True))
       self.style = nn.Sequential(*layers)


       in_channel = self.channels[4]
       for i in range(self.num_layers):
            res = (i + 5) // 2
            shape = [1,1,2 **res, 2** res]
            self.noise.register_buffer(f'noise_{i}', torch.randn(*shape))

        
       for i in range(3,self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyleConv(in_channel, out_channel, 3, style_dim,upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyleConv(out_channel, out_channel, 3, style_dim,upsample=False, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))
            in_channel = out_channel
        
       self.n_latent = self.log_size * 2 - 2

    def mean_latent(self, n_latent):
        latent = torch.randn(n_latent, self.style_dim)
        latent = self.style(latent)
        latent = latent.mean(0, keepdim=True)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(self, styles, return_latents=False, inject_index=1, truncation=1, truncation_latent=None, input_is_latent=False, noise=None, randomize_noise=True):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
        
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noise, f'noise_{i}') for i in range(self.num_layers)]
        
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + (style - truncation_latent) * truncation)
            styles = style_t
        
        if len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index == None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)
        
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        img = skip

        if return_latents:
            return img, latent
        else:
            return img, None    


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.size = size
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        convs = [ConvLayer(3, self.channels[size], 1)]
        log_size = int(math.log(self.size, 2))
        in_channel = self.channels[size]

        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, 3, blur_kernel=blur_kernel))
            in_channel = out_channel
        
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.stddev_feat = 1
        self.final_conv = ConvLayer(in_channel + 1, self.channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(self.channels[4] * 4 * 4, self.channels[4]),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(self.channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, channel, height, width)
        stddev = torch.sqrt(stddev.var(dim=0, unbiased=False) + 1e-8)
        stddev = stddev.mean(dim=[1, 2, 3], keepdim=True)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out