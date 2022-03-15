from PIL import Image
class ModelInterface:
    def __init__(self):
        self.model = self.load_pretrained_model()
    
    def load_pretrained_model(self)->None:
        pass

    def forward(self, image1, image2)->Image:
        pass