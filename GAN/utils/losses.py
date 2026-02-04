import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from typing import Optional

from .proGAN_DG import Discriminator
import torch.nn as nn

from torchvision.models.vgg import vgg16

class GANLoss:
    '''
    base GAN loss 상속해서 쓴다.
    '''
    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the discriminator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the generator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("gen_loss method has not been implemented")
    
class WganGP(GANLoss):
    """
    WGAN에서 나온 GAN 학습 불안정/mode collapse 방지용. discriminator를 EMD로 만들고, Lipschitz-1 조건 만족을 위해 gradient penalty라는 제약 조건을 준다.
    """

    def __init__(self, drift: float = 0.001) -> None:
        self.drift = drift

    @staticmethod
    def _gradient_penalty(
        dis: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        reg_lambda: float = 10,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        private helper for calculating the gradient penalty
        Args:
            dis: the discriminator used for computing the penalty
            real_samples: real samples
            fake_samples: fake samples
            depth: current depth in the optimization
            alpha: current alpha for fade-in
            reg_lambda: regularisation lambda

        Returns: computed gradient penalty
        """
        batch_size = real_samples.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(real_samples.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        # forward pass
        if labels is not None:
            assert dis.conditional, "labels passed to an unconditional discriminator"
            op = dis(merged, depth, alpha, labels)
        else:
            op = dis(merged, depth, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=op,
            inputs=merged,
            grad_outputs=torch.ones_like(op),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            real_scores = discriminator(real_samples, depth, alpha, labels)
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            real_scores = discriminator(real_samples, depth, alpha)
            fake_scores = discriminator(fake_samples, depth, alpha)
        loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self.drift * torch.mean(real_scores ** 2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(
            discriminator, real_samples, fake_samples, depth, alpha, labels=labels
        )
        loss += gp

        return loss

    def gen_loss(
        self,
        discriminator: Discriminator,
        _: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if labels is not None:
            assert discriminator.conditional, "labels passed to an unconditional dis"
            fake_scores = discriminator(fake_samples, depth, alpha, labels)
        else:
            fake_scores = discriminator(fake_samples, depth, alpha)
        return -torch.mean(fake_scores)
    
    
    
class CycleGANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss




class SRGAN_GeneratorLoss(nn.Module):
    def __init__(self):
        super(SRGAN_GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = SRGAN_TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class SRGAN_TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(SRGAN_TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]