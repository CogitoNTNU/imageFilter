from model_interface import ModelInterface
from PIL import Image
from os.path import exists
import torch
from torchvision import transforms
from modules import*


class TrinedModel(ModelInterface):
    def __init__(self):
        self.model_path = "Martin.model"
        super().__init__()

        self.transform = transforms.Compose([transforms.Resize((512, 640)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.encoder = Encoder(blocks=12)  # IDEAL BLOCKS ARE (depending on inclusion of 1 conv before output)5/6, 11/12, 19/20,
        self.adain = AdaIN()

    def load_pretrained_model(self)->None:
        if exists(self.model_path):
            self.decoder = torch.load(self.model_path)
        else:
            raise Exception(f"File {self.model_path} could not be found.")

    def forward(self, style_img, content_img)->Image:
        style_img = style_img.convert('RGB')
        content_img = content_img.convert('RGB')
        try:
            style_img = self.transform(style_img)
            content_img = self.transform(content_img)
        except:
            raise Exception(f"Error in transform of images.")

        encoder_out_style = self.encoder(style_img)      # TODO: Pass both images through encoder in one forward pass
        encoder_out_content = self.encoder(content_img)
        adain_out = self.adain(encoder_out_content, encoder_out_style)
        decoder_out = self.decoder(adain_out)

        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])

        img = decoder_out * STD[:, None, None] + MEAN[:, None, None]
        img = transforms.ToPILImage(img.permute(1,2,0))

        return img

        


    
    
