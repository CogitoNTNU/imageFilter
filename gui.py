from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
#from dummy_model import DummyModel
from trined_model import TrinedModel

class StyleTransferGUI():
    def __init__(self, model, width=1100, height=400):
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
        self.image_size_output = (330,330)
        self.upload_b1 = tk.Button(self.root, text="Upload style image", width=30, command = lambda:self.upload_im_1())
        self.upload_b1.pack()
        self.upload_b2  = tk.Button(self.root, text="Upload content image", width=30, command = lambda:self.upload_im_2())
        self.upload_b2.pack()
        self.switch_input_b  = tk.Button(self.root, text="Switch input", width=30, command = lambda:self.switch_image())
        self.switch_input_b.pack()
        self.run_text = tk.StringVar()
        self.error_message = "Upload images first then RUN!"
        self.default_message = "RUN!"
        self.run_text.set(self.default_message)
        self.run_b = tk.Button(self.root, textvariable=self.run_text, width=30, command = lambda:self.combine_images())
        self.run_b.pack()
        self.style_txt = tk.Label(self.root, text="Style",font=("Helvetica", 18))
        self.style_txt.place(x=120,y=self.height*(6/7))
        self.content_txt = tk.Label(self.root, text="Content",font=("Helvetica", 18))
        self.content_txt.place(x=self.width-120-20,y=self.height*(6/7))
        self.output_txt = tk.Label(self.root, text="This is art!",font=("Helvetica", 18))

        self.model = model
        
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
            self.canvas.create_image(self.width/2,170 ,anchor=tk.CENTER, image=self.tk_imO)    
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
            self.run_text.set(self.error_message)
        else:
            self.run_text.set(self.default_message)
            self.output_image = model.forward(self.image1, self.image2).resize(self.image_size_output)

            self.output_txt.place(x=self.width/2-20,y=self.height*(6/7))
            self.update()
    def run(self):
        self.root.mainloop()
if __name__=="__main__":
    model = TrinedModel() #DummyModel()
    gui = StyleTransferGUI(model)
    gui.run()
