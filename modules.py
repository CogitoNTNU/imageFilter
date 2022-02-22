from typing import Callable, List, Optional

import torch
from torch import nn
from torchvision.datasets.folder import default_loader
from torchvision.models import VGG, vgg19
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, vggtype: Callable[[bool], VGG] = vgg19, blocks: int = 12):
        super().__init__()
        self.features = Encoder._feature_extractor(vggtype, blocks)

    @staticmethod
    def _feature_extractor(vggType: Callable[[bool], VGG] = vgg19, blocks: int = 12):
        blocks = slice(12) if blocks is None else slice(blocks + 1)
        return vggType(True).features[blocks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


if __name__ == '__main__':

    img_path = "starry_night_full.jpg"
    img = default_loader(img_path)
    img = TF.to_tensor(img).unsqueeze_(dim=0)
    encoder = Encoder()

    with torch.no_grad():
        output = encoder(img)  # type: torch.Tensor

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    for y in range(5):
        for x in range(5):
            ax[x, y].imshow(output[0][y*5 + x].detach().numpy())

    plt.show()