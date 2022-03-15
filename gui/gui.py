from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from dummy_model import DummyModel
class StyleTransferGUI():
    def __init__(self, model, width=800, height=600):
        self.image1 = None
        self.image2 = None
        self.output_image = None

        self.tk_im1 = None
        self.tk_im2 = None
        self.tk_output_im = None

        self.root = tk.Tk()
        self.height = height
        self.width = width
        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()
        self.image_size = (300,300)
        self.upload_b1 = tk.Button(self.root, text="Upload image 1", width=30, command = lambda:self.upload_im_1())
        self.upload_b1.pack()
        self.upload_b2  = tk.Button(self.root, text="Upload image 2", width=30, command = lambda:self.upload_im_2())
        self.upload_b2.pack()
        self.switch_input_b  = tk.Button(self.root, text="Switch input", width=30, command = lambda:self.switch_image())
        self.switch_input_b.pack()
        self.run_b = tk.Button(self.root, text="RUN!", width=30, command = lambda:self.combine_images())
        self.run_b.pack()
        self.model = model
        self.error_text = None
    def switch_image(self):
        self.image1, self.image2 = self.image2, self.image1
        self.update()

    def update(self):
        if self.image1 is not None:
            self.tk_im1 = ImageTk.PhotoImage(self.image1)     
            self.canvas.create_image(20,20, anchor=tk.NW, image=self.tk_im1)    
            self.canvas.image1 = self.tk_im1
        if self.image2 is not None:
            self.tk_im2= ImageTk.PhotoImage(self.image2)     
            self.canvas.create_image(self.width-20, 20, anchor=tk.NE, image=self.tk_im2)    
            self.canvas.image2 = self.tk_im2
        if self.output_image is not None:
            self.tk_imO = ImageTk.PhotoImage(self.output_image)
            self.canvas.create_image(self.height/2, self.width/2, anchor=tk.CENTER, image=self.tk_imO)    
            self.canvas.imageO = self.tk_imO
        
    def upload_im_1(self):
        self.image1 = self.upload_image()
        self.update()
    
    def upload_im_2(self):
        self.image2 = self.upload_image()
        self.update()
    
    def upload_image(self):
        f_types = [("Jpg", '*.jpg'), ("Png", '*.png')]
        file = filedialog.askopenfilename(filetypes=f_types)
        with Image.open(file) as im:
            resized = im.resize(self.image_size)
        return resized
    
    def combine_images(self):
        if not self.image1 or not self.image2:
            self.error_text = "Please upload images"
        else:
            self.output_image = model.forward(self.image1, self.image2)
            self.update()
    def run(self):
        self.root.mainloop()
if __name__=="__main__":
    model = DummyModel()
    gui = StyleTransferGUI(model)
    gui.run()
