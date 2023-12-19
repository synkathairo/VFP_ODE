
import cv2

import torch

from fastai.vision.all import show_images

#visualize using fastai show_images function
def visualize(batch:torch.tensor, 
              example: int, 
              window_size: int): 
    #batch.shape = num_frames, batch_size, frame_size, frame_size 
    show_images(batch[0:window_size, example, :, :].detach().numpy())

#saving frames of a video: 
def save_frames(batch:torch.tensor, 
              example: int, 
              window_size: int, 
              name: str = 'mnist'): 

        for frame in range(window_size): 
            # print(batch.shape)
            save_img = batch[example, frame,:, :]
            # print(torch.max(save_img))
            save_img = batch[example, frame, :, :]#.detach().numpy()
            save_img = (255/(torch.max(save_img)-torch.min(save_img)))*save_img
            save_img = save_img.detach().numpy()
            cv2.imwrite(
                f'../torchdiffeq/images/{name}{frame}.png',
                save_img) 
        print('images saved')
