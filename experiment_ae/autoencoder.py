# coding: utf-8
from typing import Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#####################
###### MODULES ######
#####################
class Encoder(nn.Module):
    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)
        fcs = [nn.Linear(np.prod(img_size), width)]
        if depth > 1:
            fcs += [nn.Linear(width, width) for _ in range(depth - 1)]
        self.fcs = nn.ModuleList(fcs)
        self.fc_out = nn.Linear(width, latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.flatten(x)
        a = []
        for fc in self.fcs:
            # residual = x
            h = fc(h)
            h = F.relu(h)
            # x += residual
            a.append(h)

        z = self.fc_out(h)
        return z, a


class Decoder(nn.Module):
    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        fcs = [nn.Linear(latent_size, width)]
        if depth > 1:
            fcs += [nn.Linear(width, width) for _ in range(depth - 1)]
        self.fcs = nn.ModuleList(fcs)
        self.fc_out = nn.Linear(width, np.prod(img_size))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=img_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = z
        a = []
        for fc in self.fcs:
            # residual = x
            h = fc(h)
            h = F.relu(h)
            # x += residual
            a.append(h)

        h = self.fc_out(h)
        x = torch.sigmoid(h)
        x = self.unflatten(x)
        return x, a


class ConvEncoder(nn.Module):
    receptive_fields = {
        (28, 28): 128 * 5 * 5, 
        (64, 64): 128 * 9 * 9, 
        # (218, 178): 128 * 29 * 24
    }

    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=img_size[0],
                    out_channels=32,
                    kernel_size=4,
                    stride=2,
                    padding=2,
                ),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2
                ),
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2
                ),
            ]
        )
        self.flatten = nn.Flatten(start_dim=1)
        receptive_field = self.receptive_fields[img_size[1:]]

        widths = [receptive_field] + [width] * (depth - 1) + [latent_size]
        fcs = [nn.Linear(h_in, h_out) for h_in, h_out in zip(widths[:-1], widths[1:])]
        self.fcs = nn.ModuleList(fcs[:-1])
        self.fc_out = fcs[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        a = []

        h = x
        for conv_layer in self.convs:
            h = conv_layer(h)
            h = F.relu(h)
            a.append(h)

        h = self.flatten(h)
        for fc in self.fcs:
            h = fc(h)
            h = F.relu(h)
            a.append(h)

        z = self.fc_out(h)
        return z, a


class ConvDecoder(nn.Module):
    available_output_sizes = {
        (28, 28): [(5, 5), (8, 8), (15, 15), (28, 28)],
        (64, 64): [(9, 9), (17, 17), (33, 33), (64, 64)],
        # (218, 178): [(29, 24), (56, 46), (110, 90), (218, 178)]
    }

    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        receptive_field = ConvEncoder.receptive_fields[img_size[1:]]
        widths = [latent_size] + [width] * (depth - 1) + [receptive_field]
        fcs = [nn.Linear(h_in, h_out) for h_in, h_out in zip(widths[:-1], widths[1:])]
        self.fcs = nn.ModuleList(fcs)

        self.output_sizes = self.available_output_sizes[img_size[1:]]
        unflattened_size = (128,) + self.output_sizes[0]
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflattened_size)
        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=2
                ),
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=2
                ),
            ]
        )
        self.conv_out = nn.ConvTranspose2d(
            in_channels=32, out_channels=img_size[0], kernel_size=4, stride=2, padding=2
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        a = []
        h = z
        for fc in self.fcs:
            h = fc(h)
            h = F.relu(h)
            a.append(h)

        h = self.unflatten(h)
        for conv_layer, output_size in zip(self.convs, self.output_sizes[1:-1]):
            h = conv_layer(h, output_size=output_size)
            h = F.relu(h)
            a.append(h)

        h = self.conv_out(h, output_size=self.output_sizes[-1])
        x = torch.sigmoid(h)

        return x, a


####################
###### MODELS ######
####################
class AE(nn.Module):
    """Auto-encoder"""

    encoder_cls = Encoder
    decoder_cls = Decoder

    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        self.encoder = self.encoder_cls(
            width=width, depth=depth, latent_size=latent_size, img_size=img_size
        )
        self.decoder = self.decoder_cls(
            width=width, depth=depth, latent_size=latent_size, img_size=img_size
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        z, a_enc = self.encoder(x)
        x_hat, a_dec = self.decoder(z)

        A = a_enc + [z] + a_dec

        return dict(
            x_hat=x_hat,
            A=A,
            z=z
        )

    def encode(self, x: torch.Tensor, mean_only: bool = False) -> torch.Tensor:
        z, _ = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.decoder(z)
        return x_hat

    def reconstruct(self, x: torch.Tensor, mean_only: bool = False) -> torch.Tensor:
        z = self.encode(x, mean_only)
        x_hat = self.decode(z)

        return x_hat


class DAE(AE):
    """De-noising Auto-encoder"""

    def __init__(
        self,
        noise_scale: float = 0.2,
        *args, **kwargs
    ) -> None:
        super().__init__(
            *args, **kwargs
        )

        self.noise_scale = noise_scale

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x_noisy = x + self.noise_scale * torch.normal(0.0, 1.0, size=x.size()).to(
            x.device
        ).type_as(x)
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
        return x_noisy

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        if self.training:
            x = self.add_noise(x)

        z, a_enc = self.encoder(x)
        x_hat, a_dec = self.decoder(z)

        A = a_enc + a_dec

        return dict(
            x_hat=x_hat, 
            A=A,
            z=z
        )


class VAE(nn.Module):
    """Variational Auto-encoder
    
    The encoding capacity can be regulated according to [1] to prevent posterior collapse. 
    The capacity of the kl divergence can be controlled and set to a desired amount of nats. 

    [1] C. P. Burgess et al., “Understanding disentangling in $\beta$-VAE,” arXiv:1804.03599 [cs, stat], Apr. 2018, Accessed: Feb. 28, 2020. [Online]. Available: http://arxiv.org/abs/1804.03599

    """

    encoder_cls = Encoder
    decoder_cls = Decoder

    def __init__(
        self,
        width: int = 256,
        depth: int = 1,
        latent_size: int = 10,
        img_size: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        self.encoder = self.encoder_cls(
            width=width, depth=depth, latent_size=latent_size * 2, img_size=img_size
        )
        self.decoder = self.decoder_cls(
            width=width, depth=depth, latent_size=latent_size, img_size=img_size
        )

    def reparameterize(self, loc: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar / 2)
        z = loc + std * torch.randn_like(std)
        return z

    def kld(self, loc: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kld = -0.5 * torch.sum(1 + logvar - loc.pow(2) - logvar.exp()) / loc.shape[0]

        return kld

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        h, a_enc = self.encoder(x)
        loc, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(loc, logvar)
        x_hat, a_dec = self.decoder(z)

        kld = self.kld(loc, logvar)

        A = a_enc + a_dec

        return dict(
            x_hat=x_hat, 
            A=A, 
            kld=kld, 
            z=z, 
            loc=loc,
            logvar=logvar
        )

    def encode(self, x: torch.Tensor, mean_only: bool = False) -> torch.Tensor:
        h, _ = self.encoder(x)
        loc, logvar = h.chunk(2, dim=-1)
        if mean_only:
            return loc

        z = self.reparameterize(loc, logvar)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.decoder(z)
        return x_hat

    def reconstruct(self, x: torch.Tensor, mean_only: bool = False) -> torch.Tensor:
        z = self.encode(x, mean_only)
        x_hat = self.decode(z)

        return x_hat


class CAE(AE):
    """Convolutional Auto-encoder"""

    encoder_cls = ConvEncoder
    decoder_cls = ConvDecoder


class CDAE(DAE, CAE):
    """Convolutional De-noising Auto-encoder"""

    pass


class CVAE(VAE):
    """Convolutional Auto-encoder"""

    encoder_cls = ConvEncoder
    decoder_cls = ConvDecoder