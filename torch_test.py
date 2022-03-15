import torch
x = torch.rand(2,3)
assert x.size(dim=1)==3
print("Torch is working!!")

import PIL
print('PIL',PIL.__version__)