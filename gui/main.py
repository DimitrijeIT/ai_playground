# The ultimate introduction to modern GUIs in Python [ with tkinter ]
# https://www.youtube.com/watch?si=A89gpiLfBfAUpvuj&t=60666&v=mop6g-c5HEY&feature=youtu.be

import customtkinter as ctk
from image_widges import *
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
from manu import *

class App(ctk.CTk):
    def __init__(self):

        #setup 
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x600') #default size
        self.title('Photo Editor')
        self.minsize(800, 500)
        self.init_parementers()

        #layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2, uniform= 'a')
        self.columnconfigure(1, weight=6, uniform= 'a')

        #canvas image
        self.image_width = 0
        self.image_height = 0
        self.canvas_width = 0
        self.canvas_height = 0

        #widgets
        #Import Buttom (Frame with a buttom)
        self.image_import = ImageImport(self, self.import_image)

        #run 
        self.mainloop()
    
    def init_parementers(self):
        self.pos_vars = {
            'rotate': ctk.DoubleVar(value = ROTATE_DEFAULT),
            'zoom': ctk.DoubleVar(value = ZOOM_DEFAULT),
            'flip': ctk.StringVar(value = FLIP_OPTIONS[0])
        }
        # self.rotate_float = ctk.DoubleVar(value = ROTATE_DEFAULT)
        # self.zoom_float = ctk.DoubleVar(value = ZOOM_DEFAULT)

        self.color_vars = {
            'brightness': ctk.DoubleVar(value = BRIGHTNESS_DEAFULT),
            'grayscale': ctk.BooleanVar(value = GRAYSACELE_DEFATULT), 
            'invert': ctk.BooleanVar(value = INVERT_DEFATULT),
            'vibrance': ctk.DoubleVar(value = VIBRANCE_DEFAULT)
        }       

        self.effect_vars = {
            'blur': ctk.DoubleVar(value = BLUR_DEFAULT), 
            'contrast' : ctk.IntVar(value = CONTRAST_DEFAULT), 
            'effect': ctk.StringVar(value = EFFECT_OPTIONS[0])
        }

        #tracing
        combined_vars = list(self.pos_vars.values()) + list(self.color_vars.values()) + list(self.effect_vars.values())
        for var in combined_vars:
            var.trace('w', self.manipulate_image)

        # self.rotate_float.trace('w', self.manipulate_image)
        # self.zoom_float.trace('w', self.manipulate_image)

    def manipulate_image(self, *args):
        self.image = self.orignal

        # rotate
        if self.pos_vars['rotate'].get() != ROTATE_DEFAULT:
            self.image = self.image.rotate(self.pos_vars['rotate'].get())

        #zoom
        if self.pos_vars['zoom'].get() != ZOOM_DEFAULT:    
            self.image = ImageOps.crop(image=self.image, border=self.pos_vars['zoom'].get())

        #flip
        if self.pos_vars['flip'].get() != FLIP_OPTIONS[0]:
            if self.pos_vars['flip'].get() == 'X':
                self.image = ImageOps.mirror(self.image)
            if self.pos_vars['flip'].get() == 'Y':
                self.image = ImageOps.flip(self.image) 
            if self.pos_vars['flip'].get() == 'Both':
                self.image = ImageOps.mirror(self.image)
                self.image = ImageOps.flip(self.image)   

        #brightness and vibrance
        if self.color_vars['brightness'].get() != BRIGHTNESS_DEAFULT:        
            brigness_encancher = ImageEnhance.Brightness(self.image)
            self.image = brigness_encancher.enhance(self.color_vars['brightness'].get())

        if self.color_vars['vibrance'].get() != VIBRANCE_DEFAULT:           
            vibrance_encancher = ImageEnhance.Brightness(self.image)
            self.image = vibrance_encancher.enhance(self.color_vars['vibrance'].get())

        #color
        if self.color_vars['grayscale'].get():
            self.image = ImageOps.grayscale(self.image)

        if self.color_vars['invert'].get():
            self.image = ImageOps.invert(self.image)


        #blur and contrast 
        if self.effect_vars['blur'].get() != BLUR_DEFAULT:
            self.image = self.image.filter(ImageFilter.GaussianBlur(self.effect_vars['blur'].get()))
        if self.effect_vars['contrast'].get() != CONTRAST_DEFAULT:
            self.image = self.image.filter(ImageFilter.UnsharpMask(self.effect_vars['contrast'].get()))

        match self.effect_vars['effect']:
            case 'Emboss':self.image = self.image.filter(ImageFilter.EMBOSS)
            case 'Find edges':self.image = self.image.filter(ImageFilter.FIND_EDGES)
            case 'Contour':self.image = self.image.filter(ImageFilter.CONTOUR)
            case 'Edge enhance':self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)

            
        self.place_image()


    def import_image(self, path):
        print(path)
        self.orignal = Image.open(path)
        self.image = self.orignal
        self.image_ratio = self.image.size[0] / self.image.size[1]

        self.image_tk = ImageTk.PhotoImage(self.image)
        # self.image.show()
        
        self.image_import.grid_forget()
        self.image_output = ImageOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.close_edit)
        self.menu = Menu(self, self.pos_vars, self.color_vars, self.effect_vars, self.export_image)
    
    def close_edit(self):
        #hide the iamge and the close button
        self.image_output.grid_forget()
        self.close_button.place_forget() #called with place so place_forget()
        self.menu.grid_forget()

        # recreate the import button. Same as when firt created
        self.image_import = ImageImport(self, self.import_image)


    def resize_image(self, event):
        canvas_ratio = event.width / event.height

        self.canvas_width = event.width
        self.canvas_height = event.height

        #resize:
        if canvas_ratio > self.image_ratio: # canvas is wider then the image
            self.image_height = int(event.height)
            self.image_width = int(self.image_height * self.image_ratio)
        else: #canvas taller then the image
            self.image_width = int(event.width)
            self.image_height = int(self.image_width / self.image_ratio)
        
        self.place_image()

    def place_image(self):
        #place image
        self.image_output.delete('all')

        resized_image = self.image.resize((self.image_width, self.image_height ))
        self.image_tk = ImageTk.PhotoImage(resized_image)

        self.image_output.create_image(self.canvas_width/2, self.canvas_height/2, image = self.image_tk)

    def export_image(self, name, file, path):
        export_string = f'{path}/{name}.{file}'
        self.image.save(export_string)

App()