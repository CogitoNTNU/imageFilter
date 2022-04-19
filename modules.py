import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Callable, List, Optional

import torch
from torch import nn
from torchvision.datasets.folder import default_loader
from torchvision.models import VGG, vgg19
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    :param loss_weight: weight of the loss
    """
    def __init__(self, loss_weight: float = 1e-3):
        super().__init__()
        self.tv_loss_weight = loss_weight

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


class AdaINContentLoss(nn.MSELoss):
    def __init__(self, loss_weight: float = 1e-3):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x_features: torch.Tensor, adain_output: torch.Tensor) -> torch.Tensor:
        return super(AdaINContentLoss, self).forward(x_features, adain_output) * self.loss_weight


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

class AdaINLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_
        # lambda is a hyperparameter the dictates the relative importance of content vs style
        # the greater lambda is the more the model will try to preserve style
        # the smaller lambda is the more the model will try and preserve content
        # [See equation 11 of the Paper]

    def contentLoss(self, content_emb, output_emb):
        """ Takes 2 embedding tensors generated by vgg and finds the L2 norm
        (ie. euclidan distance) between them. [See equation 12 of the Paper]"""
        return torch.norm(content_emb-output_emb)

    def styleLoss(self, style_activations, output_activations):
        """ Takes 2 lists of activation tensors hooked from vgg layers during
        forward passes using our style image and our ouput image as inputs.
        Computes the L2 norm between each of their means and standard deviations
        and returns the sum. [See equation 13 of the Paper]"""
        mu_sum = 0
        sigma_sum = 0
        for style_act, output_act in zip(style_activations, output_activations):
            mu_norm = torch.norm(self.mu(output_act)-self.mu(style_act))
            mu_sum += mu_norm
            sigma_norm = torch.norm(self.sigma(output_act)-self.sigma(style_act))
            sigma_sum += sigma_norm
        return mu_sum + sigma_sum

    def totalLoss(self, content_emb, output_emb, style_activations, output_activations):
        """ Calculates the overall loss. [See equation 11 of the Paper]"""
        content_loss = self.contentLoss(content_emb, output_emb)
        style_loss = self.styleLoss(style_activations, output_activations)
        print(content_loss.item(), style_loss.item())
        return content_loss+self.lambda_*style_loss

    def forward(self, content_emb, output_emb, style_activations, output_activations):
        """ For caculating single image loss please pass arguments with a batch size of 1. """
        return self.totalLoss(content_emb, output_emb, style_activations, output_activations)/content_emb.shape[0]
    
    def mu(self, x):
        """ Takes a (c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(1,2))/(x.shape[1]*x.shape[2])

    def sigma(self,x):
        """ Takes a (c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(torch.sum((x.permute([1,2,0])-self.mu(x)).permute([1,2,0])**2,(1,2))/(x.shape[1]*x.shape[2]))
'''
class AdaIN(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        mean_y = y_features.mean(dim=1, keepdim=True)
        std_y = y_features.std(dim=1, keepdim=True)
        mean_x = x_features.mean(dim=1, keepdim=True)
        std_x = x_features.std(dim=1, keepdim=True)
        output = std_y * ((x_features - mean_x) / (std_x + 1e-8)) + mean_y
        return output
'''
class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""

        output = (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])
       
        return output


if __name__ == '__main__':

    content_img_path = "starry_night_full.jpg"
    style_img_path = "lofoten.jpg"
    style_img = default_loader(style_img_path)
    content_img = default_loader(content_img_path)

    style_img = TF.resize(TF.to_tensor(style_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used
    content_img = TF.resize(TF.to_tensor(content_img), (512, 640))  # TODO: Automatically find the needed size based on VGG-blocks used

    style_img.unsqueeze_(dim=0)
    content_img.unsqueeze_(dim=0)

    style_img = style_img.to(device)
    content_img = content_img.to(device)

    encoder = Encoder(blocks=12)  # IDEAL BLOCKS ARE (depending on inclusion of 1 conv before output)5/6, 11/12, 19/20,
    adain = AdaIN()
    decoder = Decoder(blocks=12)

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False


    encoder = encoder.to(device)
    adain = adain.to(device)
    decoder = decoder.to(device)


    style_loss = FeatureLoss(encoder, loss_weight=0.1)
    image_loss = ImageLoss(encoder.features[:6], pixel_loss_weight=2, feature_loss_weight=0.5)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    for epoch in range(100):
        encoder_out_style = encoder(style_img)      # TODO: Pass both images through encoder in one forward pass
        encoder_out_content = encoder(content_img)
        adain_out = adain(encoder_out_content, encoder_out_style)
        decoder_out = decoder(adain_out)
        loss = style_loss(decoder_out, style_img) + image_loss(decoder_out, content_img)
        print(f"Epoch {epoch}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 99 == 0:
            plt.imshow(decoder_out.detach().cpu().squeeze(dim=0).numpy().transpose(1, 2, 0))
            plt.show()


    
    
    # with torch.no_grad():
    #     output = encoder(img)  # type: torch.Tensor
    #
    # fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    # for y in range(5):
    #     for x in range(5):
    #         ax[x, y].imshow(output[0][y*5 + x].detach().numpy())
    #
    # plt.show()