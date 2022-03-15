from model_interface import ModelInterface
from PIL import Image
class DummyModel(ModelInterface):
    def __init__(self):
        super().__init__()

    def forward(self, _1, _2)->Image:
        """
        Returns a PIL Image in the form of a gradient
        """
        width = 300
        height = 300
        colour1 = (255,0,0)
        colour2 = (0,255,0)
        base = Image.new('RGB', (width, height), colour1)
        top = Image.new('RGB', (width, height), colour2)
        mask = Image.new('L', (width, height))
        mask_data = []
        for y in range(height):
            mask_data.extend([int(255 * (y / height))] * width)
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)
        return base
    
    
