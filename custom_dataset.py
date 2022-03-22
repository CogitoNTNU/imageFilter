import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if image.size()[0] == 1: image= image[0].repeat(3, 1, 1)
        return image
    


if __name__ == '__main__':
    dataset = CustomImageDataset('.\\style_images\\test_samples',transform = transforms.Resize((512, 640)))

    bs = 10
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    iterator = iter(dataloader)
    for i in range(10):
        images = next(iterator)
        print(f"Feature batch shape: {images.size()}")

        fig, axs = plt.subplots(bs,1,figsize=(5, 5*bs))
        for i in range(bs):
            img = images[i].permute(1,2,0).squeeze()
            axs[i].axis("off")
            axs[i].imshow(img)
        plt.show()



