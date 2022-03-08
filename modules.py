from typing import Callable, List, Optional

import torch
from torch import nn
from torchvision.datasets.folder import default_loader
from torchvision.models import VGG, vgg19
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class Sequential(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.add_modules(**kwargs)

    def add_modules(self, **kwargs):
        for key, value in kwargs.items():
            self.add_module(key, value)


class UpsampleBlock(Sequential):
    def __init__(self, in_c, out_c, upscale_factor: int = 2, act: str = "relu", dropout: float = 0.0):
        super().__init__()
        if act == "relu":
            act_fn = nn.ReLU()
        elif act == "leaky":
            act_fn = nn.LeakyReLU(0.2)
        elif act == "prelu":
            act_fn = nn.PReLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        self.add_modules(
            conv=nn.Conv2d(in_c, out_c * (upscale_factor ** 2), 3, 1, 1, padding_mode='replicate'),
            ps=nn.PixelShuffle(upscale_factor),
            act=act_fn
        )
        if 0 < dropout < 1:
            self.add_module('dropout', nn.Dropout2d(dropout))

    def __repr__(self):
        return f"UpsampleBlock({self.conv.in_channels}, {self.conv.out_channels}, {self.ps.upscale_factor})"

class Encoder(nn.Module):

    def __init__(self, vggtype: Callable[[bool], VGG] = vgg19, blocks: int = 20):
        super().__init__()
        self.features = Encoder._feature_extractor(vggtype, blocks)
        print("Encoder:", self.features)

    @staticmethod
    def _feature_extractor(vggType: Callable[[bool], VGG] = vgg19, blocks: int = 20):
        blocks = slice(blocks + 1)
        features = vggType(True).features[blocks]
        feature_blocks = []
        current_feature_block = []
        for i, layer in enumerate(features):
            if isinstance(layer, nn.MaxPool2d):
                current_feature_block.append(layer)
                feature_blocks.append(nn.Sequential(*current_feature_block, layer))
                current_feature_block = []
            else:
                current_feature_block.append(layer)
        if current_feature_block:
            feature_blocks.append(nn.Sequential(*current_feature_block))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class Decoder(nn.Module):

    def __init__(self, vggtype: Callable[[bool], VGG] = vgg19, blocks: int = 20):
        super().__init__()
        self.features = Decoder._feature_extractor(vggtype, blocks)
        print("Decoder:", self.features)

    @staticmethod
    def _feature_extractor(vggType: Callable[[bool], VGG] = vgg19, blocks: int = 20):
        blocks = slice(blocks + 1)
        features = vggType(True).features[blocks]
        selected_blocks = []
        last_in, last_out = 0, 0
        for i, layer in enumerate(reversed(features[1:])):
            if isinstance(layer, nn.Conv2d):
                last_in, last_out = layer.in_channels, layer.out_channels
                selected_blocks.append(nn.Conv2d(last_out, last_in, 3, 1, 1, padding_mode='replicate'))
            elif isinstance(layer, nn.MaxPool2d):
                selected_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            elif isinstance(layer, nn.ReLU):
                selected_blocks.append(nn.ReLU())
            elif isinstance(layer, nn.LeakyReLU):
                selected_blocks.append(nn.LeakyReLU(layer.negative_slope))
            elif isinstance(layer, nn.PReLU):
                selected_blocks.append(nn.PReLU(layer.num_parameters, init=layer.init))
        selected_blocks.append(nn.Conv2d(last_out, 3, 3, 1, 1, padding_mode='replicate'))
        return nn.Sequential(*selected_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TVLoss(nn.Module):
    """
    Total Variation regularizer (reduces high frequency structures)
    :param tv_loss_weight: weight of the loss
    """
    def __init__(self, tv_loss_weight: float = 1e-3):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        return t.numel()

    def __repr__(self):
        return f"TVLoss({self.tv_loss_weight})"


class FeatureLoss(nn.MSELoss):
    def __init__(self, encoder: nn.Module, loss_weight: float = 1e-3):
        super().__init__()
        self.encoder = encoder
        self.loss_weight = loss_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return super(FeatureLoss, self).forward(self.encoder(x), self.encoder(y)) * self.loss_weight


class PixelLoss(nn.MSELoss):
    def __init__(self, loss_weight: float = 1e-3):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return super(PixelLoss, self).forward(x, y) * self.loss_weight


class ImageLoss(nn.Module):

    def __init__(self, encoder: nn.Module,
                 pixel_loss_weight: float = 1e-3, feature_loss_weight: float = 1e-3, tv_loss_weight: float = 1e-3):
        super().__init__()
        self.pixel_loss = PixelLoss(pixel_loss_weight)
        self.feature_loss_fn = FeatureLoss(encoder, feature_loss_weight)
        self.tv_loss_fn = TVLoss(tv_loss_weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.pixel_loss(x, y) + self.feature_loss_fn(x, y) + self.tv_loss_fn(x)


class AdaIN(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        mean_y = y_features.mean(dim=1, keepdim=True)
        std_y = y_features.std(dim=1, keepdim=True)
        mean_x = x_features.mean(dim=1, keepdim=True)
        std_x = x_features.std(dim=1, keepdim=True)
        print(mean_y.shape, std_y.shape, mean_x.shape, std_x.shape)
        adain = std_y * ((x_features - mean_x) / (std_x + 1e-8)) + mean_y
        print(adain.shape)
        return adain




if __name__ == '__main__':

    content_img_path = "lofoten.jpg"
    style_img_path = "starry_night_full.jpg"
    style_img = default_loader(style_img_path)
    content_img = default_loader(content_img_path)

    style_img = TF.resize(TF.to_tensor(style_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used
    content_img = TF.resize(TF.to_tensor(content_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used

    style_img.unsqueeze_(dim=0)
    content_img.unsqueeze_(dim=0)

    encoder = Encoder()

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    adain = AdaIN()
    decoder = Decoder()

    style_loss = FeatureLoss(encoder)
    tv_loss = TVLoss()
    pixel_loss = PixelLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    for epoch in range(100):
        encoder_out_style = encoder(style_img)
        encoder_out_content = encoder(content_img)
        print(encoder_out_style.shape, encoder_out_content.shape)
        adain_out = adain(encoder_out_content, encoder_out_style)
        decoder_out = decoder(adain_out)
        loss = style_loss(decoder_out, style_img) + tv_loss(decoder_out) + pixel_loss(decoder_out, content_img)
        print(f"Epoch {epoch}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            plt.imshow(decoder_out.detach().squeeze(dim=0).numpy().transpose(1, 2, 0))
            plt.show()


    #
    #
    # with torch.no_grad():
    #     output = encoder(img)  # type: torch.Tensor
    #
    # fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    # for y in range(5):
    #     for x in range(5):
    #         ax[x, y].imshow(output[0][y*5 + x].detach().numpy())
    #
    # plt.show()