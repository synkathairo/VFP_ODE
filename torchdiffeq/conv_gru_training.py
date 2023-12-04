import numpy as np 
import argparse
import matplotlib.pyplot as plt
import cv2

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchdiffeq import odeint
from base_conv_gru import ConvGRUCell

from fastai.vision.all import show_images

from datasets.moving_mnist import build, MyCollate

parser = argparse.ArgumentParser('test')

parser.add_argument('--path', default='/Users/ssarch/Documents/acads/sem-1/Image&video_process/proj/torchdiffeq/data')
parser.add_argument('--frame_size', type=int, default=64)
parser.add_argument('--digit_size', type=int, default=28)
parser.add_argument('--step_length', type=float, default=0.5)
parser.add_argument('--nframes', type=int, default=16)
parser.add_argument('--training', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)


args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

args.train = True

mmnist_train = build(training=True, args=args)

train_loader = torch.utils.data.DataLoader(
         dataset=mmnist_train, batch_size=args.batch_size, num_workers=args.workers,
         pin_memory=True, collate_fn = MyCollate(nframes=args.nframes, frame_size=args.frame_size))


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
                f'/Users/ssarch/Documents/acads/sem-1/Image&video_process/proj/torchdiffeq/images/{name}{frame}.png',
                save_img) 
        print('images saved')
# /Users/ssarch/Documents/acads/sem-1/Image&video_process/proj/torchdiffeq/plots/bouncing_ball.png

############################################

class Model(nn.Module): 
    
    def __init__(self,
                 inchannels: int): 
        super(Model, self).__init__()
        
        self.inchannels = inchannels
        self.res_block = nn.Sequential(nn.Conv2d(inchannels+1, 16, 3, padding='same'), 
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(num_features=16), 
                                 nn.Conv2d(16, inchannels, 3, padding='same' ), 
                                 nn.ReLU()
                            ) 

    def forward(self, t, inp): 
       reduce_out = False
       if inp.shape[-3]> self.inchannels: 
           inp = inp.unsqueeze(-3).repeat(1,self.inchannels, 1,1)
           reduce_out = True

       a = torch.ones(inp.shape)*t
       inp = torch.cat((a, inp), dim=1)
        
       out = self.res_block(inp) # out.shape = batch_size, num_channels, frame_size, frame_size
       
       if reduce_out: 
           return out.squeeze(-3)
       return out 
    
class ConvEncoder(nn.Module): 
    
    def __init__(self, 
                 input_size: tuple = (64, 64), 
                 input_dim: int = 1, 
                 hidden_dim: int = 3, 
                 kernel_size: tuple = (3,3), 
                ): 
        super(ConvEncoder, self).__init__()
        
        self.conv_gru = ConvGRUCell(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim, 
                           kernel_size=kernel_size, bias=True) 

        self.conv_hid2out = nn.Conv2d(hidden_dim,input_dim, input_dim, padding='same')

        
    def forward(self, 
                inp: torch.tensor, 
                hidden: torch.tensor = None): 
            
        #inp.shape = batch, time, frame_size, frame_size 
        #hidden.shape = batch, 1, frame_size, frame_size 

        
        hidd = self.conv_gru(inp, hidden) 
        out = self.conv_hid2out(hidd)
        out = F.relu(out)
        return out, hidd

############################################
if __name__ == '__main__': 
    
    func = Model(inchannels=1)
    optimizer = torch.optim.RMSprop(func.parameters(), lr=1e-3)
    conv_enc = ConvEncoder()

    for n, i in enumerate(train_loader): 

        # visualize(i, 0, 5)
        # save_frames(i, 0, 5)
        # print(i.shape)
        
        #i.shape = batch, time, frame_size, frame_size
        inp = i[:, 0, :, :]#.unsqueeze(1) #batch_size, time(=1), frame_size, frame_size
        t  = torch.linspace(0, 1, i.shape[1]-1)#[:8]
        trg = i[:, 1:, :, :] #batch, time, frame_size, frame_size

        # print(inp.shape, t.shape, trg.shape)

        optimizer.zero_grad()
        # pred = odeint(func, inp, t).to(device) #pred.shape = time, batch_size, frame_size, frame_size
        h_curr = torch.zeros(i[:, 0, :, :].unsqueeze(1).shape).repeat(1,3,1,1)
        pred = i[:, 0, :, :].unsqueeze(1)
        for time in range(i.shape[1]-1): 
            inp = i[:,time, :, : ].unsqueeze(1)
            out, h_next = conv_enc(inp, h_curr)
            h_curr = h_next 
            pred = torch.cat((pred,out), dim=1) 
        # print(i.shape, inp.shape, trg.shape, pred.shape)

        # pred = pred.squeeze(2)
        # pred = pred.permute(1,0,2,3)
        pred = pred[:, 1:, :, :]

        #mse_loss: 
        loss = torch.mean(torch.abs(pred-trg))
        print(loss)        
        loss.backward()
        optimizer.step()

        if n > 10: 
            save_frames(batch=pred, example=2, window_size=pred.shape[1], name="pred")
            save_frames(batch=trg, example=2, window_size=trg.shape[1], name="trg")
            break
