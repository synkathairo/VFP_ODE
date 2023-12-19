
import random
import numpy as np 
import argparse
import cv2

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchdiffeq import odeint
from base_conv_gru import ConvGRUCell

from fastai.vision.all import show_images

from models import Model, ConvEncoder, ConvDecoder
from utils_aneesh import visualize

import pytorch_warmup as warmup

import wandb 

#setting seeds: 
MANUAL_SEED = 3407
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True

from datasets.moving_mnist import build, MyCollate

parser = argparse.ArgumentParser('test')

parser.add_argument('--path', default='../torchdiffeq/data')
parser.add_argument('--frame_size', type=int, default=64)
parser.add_argument('--digit_size', type=int, default=28)
parser.add_argument('--step_length', type=float, default=0.5)
parser.add_argument('--nframes', type=int, default=32)
parser.add_argument('--training', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=0)

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--warmup_period', type=int, default=50)

args = parser.parse_args()

'''
wandb.init(config=args, project="vid_ode")
wandb.run.name = "Conv_gru+odeint+decoder"
wandb.config.update(args)
config = wandb.config 
wandb.run.save()
'''

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
            save_img2 = batch[example, frame, :, :].detach().numpy()
            cv2.imwrite(
                f'../torchdiffeq/images/{name}{frame}_non_proc.png',
                save_img2) 
            save_img = (255/(torch.max(save_img)-torch.min(save_img)))*save_img
            save_img = save_img.detach().numpy()
            cv2.imwrite(
                f'../torchdiffeq/images/{name}{frame}.png',
                save_img) 
        print('images saved')

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


############################################

def main(): 
    
	args.train = True

	mmnist_train = build(training=True, args=args)

	func = Model(inchannels=3)#.to(args.rank)

	conv_enc = ConvEncoder()

	conv_dec = ConvDecoder(inp_dim=3)

	optimizer = torch.optim.Adamax(func.parameters(), lr=args.learning_rate)

	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=args.warmup_period)

	train_loader = torch.utils.data.DataLoader(
			dataset=mmnist_train, batch_size=args.batch_size, num_workers=args.workers,
			pin_memory=True,  collate_fn = MyCollate(nframes=args.nframes, frame_size=args.frame_size))

	epochs = args.epochs

	for epoch in range(epochs):  


		func.train()
		conv_enc.train()
		conv_dec.train()

		epoch_loss = 0 

		for n, i in enumerate(train_loader): 

            #i.shape = batch, time, frame_size, frame_size

			inp = i[:, 0, :, :]#.cuda(gpu, non_blocking=True) #batch_size, time(=1), frame_size, frame_size

			t  = torch.linspace(0, 1, i.shape[1]-1)#[:8]

			trg = i[:, 1:, :, :]#.cuda(gpu, non_blocking=True) #batch, time, frame_size, frame_size

            # pred = odeint(func, inp, t).to(device) #pred.shape = time, batch_size, frame_size, frame_size

			h_curr = torch.zeros(i[:, 0, :, :].unsqueeze(1).shape).repeat(1,3,1,1)#.cuda(gpu, non_blocking=True)
			pred = torch.ones(i.shape)[:, 0, :, :].unsqueeze(1)#.cuda(gpu, non_blocking=True)
            # pred = i[:, 0, :, :].unsqueeze(1)

			img_diff_seq = torch.zeros(i.shape)[:, 0, :, :].unsqueeze(1)#.cuda(gpu, non_blocking=True)

			for time in range(i.shape[1]-1): 

				inp = i[:,time, :, : ].unsqueeze(1)
				mean = torch.mean(inp, dim=(-1, -2)).unsqueeze(-1)
				mean = mean.unsqueeze(-1).repeat(1,1,inp.shape[2],inp.shape[3])
				inp = inp-mean

				h_next = odeint(func, h_curr, t).to(device)[-1, :, :, :] #pred.shape = time, batch_size, frame_size, frame_size
				out, h_next = conv_enc(inp, h_next)
				of, img_diff, out = conv_dec(h_next, inp)
				h_curr = h_next 
				pred = pred + mean
				pred = torch.cat((pred,out), dim=1) 
				img_diff_seq = torch.cat((img_diff_seq, img_diff), dim=1)

			pred = pred[:, :-1, :, :]
			img_diff_seq = img_diff_seq[:, 1:, :, :]

			optimizer.zero_grad()
            #mse_loss: 

			loss = torch.mean(torch.abs(pred-trg))

			id_trg = i[:, :-1, :, :] - trg
			loss_id = torch.mean(torch.abs(img_diff_seq - id_trg)) 
			loss = loss + loss_id
            # wandb.log({"iter_loss": loss})
			print(loss)        
			loss.backward()
			optimizer.step()
			with warmup_scheduler.dampening(): 
				lr_scheduler.step()


			epoch_loss +=loss

		epoch_loss = epoch_loss.div_(len(train_loader))
        # wandb.log({"epoch_loss": epoch_loss})

        # wandb.log({"pred_vid": wandb.Video(pred[1,:,:,:].unsqueeze(1).repeat(1,3,1,1).detach().numpy(), fps=2)})
        # wandb.log({"trg_vid": wandb.Video(trg[1,:,:,:].unsqueeze(1).repeat(1,3,1,1).detach().numpy(), fps=2)})
		save_frames(batch=pred, example=1, window_size=pred.shape[1], name=f"pred_epoch_{epoch}")
		save_frames(batch=trg, example=1, window_size=trg.shape[1], name=f"trg_epoch_{epoch}")

    # wandb.finish()

if __name__ == "__main__": 
	main()
