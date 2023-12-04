# dataloader for moving mnist dataset

from pathlib import Path 
import random
from random import choice

import numpy as np 

import torch 
from torchvision import transforms 

import tensorflow_datasets as tfds
from fastai.vision.all import *

from PIL import Image


# generating random trajectories:
class MovingDigits:

    def __init__(self, 
                 frame_size:int, 
                 digit_size:int, 
                 step_length:float, 
                 nframes: int, 
                 path=None, 
				 size_dataset:int=10000, 
                 training: bool=True):
            
            self.path = path
            self.frame_size = frame_size
            self.digit_size = digit_size
            self.step_length = step_length
            self.nframes = nframes
            self.size_dataset = size_dataset
            #download MNIST files: 
            if not(self.path):
                self.path = untar_data(URLs.MNIST)

            if training: 
                self.files = get_image_files(self.path/'training')
            else:
                self.files = get_image_files(self.path/'testing')
    
    def get_random_trajectory(self): 
        "Generate a trajectory"
        canvas_size = self.frame_size - self.digit_size
        x, y, v_x, v_y = np.random.random(4)
        out_x, out_y = [], []

        for i in range(self.nframes):
            # Take a step along velocity.
            y += v_y * self.step_length
            x += v_x * self.step_length

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            out_x.append(x * canvas_size)
            out_y.append(y * canvas_size)

        return torch.tensor(out_x, dtype=torch.uint8), torch.tensor(out_y, dtype=torch.uint8)
    
    def generate_moving_digit(self):

        img = Image.open(choice(self.files))
        digit_image = transforms.PILToTensor()(img) #load a random image

        ##################################
        #resize digit_image to digit_size
        ##################################
        xs, ys = self.get_random_trajectory()
        frame = torch.zeros((self.nframes, 1, self.frame_size, self.frame_size))
        # print('###########################################')
        # print(frame.shape,xs, ys)
        for i, (x,y) in enumerate(zip(xs, ys)):
            frame[i, 0, y:(y+self.digit_size), x:(x+self.digit_size)] = digit_image

        return frame


    def generate_random_digits(self, digits:int=1): 
        return torch.stack([self.generate_moving_digit() for n in range(digits)]).max(dim=0)[0]
    

    def __getitem__(self, idx): 
        
        num_digits = random.randint(1,4)
        vid =  self.generate_random_digits(digits=num_digits)
        return vid

    def __len__(self): 
       return self.size_dataset
         
def build(training: bool, args): 
    
    
    #load dataset: 
    dataset = MovingDigits(frame_size=args.frame_size, 
                           digit_size=args.digit_size, 
                           step_length=args.step_length, 
                           nframes=args.nframes, 
                           training = training)            

    return dataset
                

class MyCollate: 
    
    def __init__(self, 
                 nframes: int, 
                 frame_size: int): 
            
            self.nframes = nframes
            self.frame_size = frame_size

    def __call__(self, batch): 
        
        vid_batch = torch.zeros(self.nframes, 1, self.frame_size, self.frame_size)
        for i in batch: 

            # print(i.shape)
            vid_batch = torch.cat((vid_batch, i), dim=1) 
            # print('############################')
            # print(vid_batch.shape)

        return vid_batch[:, 1:, :, :].permute(1,0,2,3)

            
        

