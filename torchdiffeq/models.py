# model definitions
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

from base_conv_gru import ConvGRUCell

class Model(nn.Module): 
    
    def __init__(self,
                 inchannels: int): 
        super(Model, self).__init__()
        
        self.inchannels = inchannels
        self.res_block = nn.Sequential(nn.Conv2d(inchannels, 16, 3, padding='same'), 
                                 nn.ReLU(), 
                                 nn.BatchNorm2d(num_features=16), 
                                 nn.Conv2d(16, inchannels, 3, padding='same' ), 
                            ) 

    def forward(self, t, inp): 
       reduce_out = False
       if inp.shape[-3]> self.inchannels: 
           inp = inp.unsqueeze(-3).repeat(1,self.inchannels, 1,1)
           reduce_out = True

       a = torch.ones(inp.shape[0], inp.shape[2], inp.shape[3])*t
       a = a.unsqueeze(1)
    #    inp = torch.cat((a, inp), dim=1)
        
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
        # out = F.relu(out)
        return out, hidd

class ConvDecoder(nn.Module): 
    
    def __init__(self, 
                 inp_dim:int, 
                 out_dim: int = 4, 
                 n_ups:int = 2): 
        super(ConvDecoder, self).__init__()
        
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.n_ups = n_ups

        self.convT = nn.ConvTranspose2d(inp_dim, out_dim//2, 3, padding=(1,1), stride=(1,1), dilation=(1,1))
        self.convT2 = nn.ConvTranspose2d(out_dim//2, out_dim, 3, padding=(1,1), stride=(1,1), dilation=(1,1))
        self.conv_mask2out = nn.Conv2d(1, 1, 3, padding='same')
        self.conv_hid2out = nn.Conv2d(2, 1, 3, padding='same')
       
    # code for wrapping from: https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L139
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        if x.is_cuda:
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        else: 
            mask = torch.ones(x.size())
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask   

    def forward(self, 
                inp: torch.tensor, 
                prev_frame: torch.tensor): 
        
        #inp.shape = batch, inp_dim, frame_size, frame_size
        out = self.convT(inp) #out.shape = batch_size, out_dim, frame_size, frame_size
        out = F.relu(out)
        out = self.convT2(out)
        mask = out[:,0, :, :].unsqueeze(1) #mask.shape = batch_size, frame_size, frame_size
        mask = F.sigmoid(mask)
        optical_flow = out[:, 0:2, :, :]#.unsqueeze(1) 
        image_diff = out[:, 3, :, :].unsqueeze(1) 

        # temp = torch.cat((optical_flow, prev_frame), dim=1)
        temp = self.warp(prev_frame, optical_flow)
        pred = mask*self.conv_mask2out(temp) + (1-mask)*image_diff

        return optical_flow, image_diff, pred 

 