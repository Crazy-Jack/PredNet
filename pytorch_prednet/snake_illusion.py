from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import hickle as hkl

import torch
from torch.utils.data import DataLoader
import numpy as np

from kitti_data import KITTI
from kitti_settings import *
from prednet import PredNet
from PIL import Image
import cv2
import sys 

# Visualization parameters
n_plot = 4 # number of plot to make (must be <= batch_size)

# Model parameters
gating_mode = 'mul'
peephole = False
lstm_tied_bias = False
nt = 10 
extrap_start_time = None
batch_size = 4

default_channels = (3, 48, 96, 192)
channel_six_layers = (3, 48, 96, 192, 384, 768)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)


MODEL_DIR = 'models/' 
model_name = 'prednet-L_0-mul-peepFalse-tbiasFalse-best'
# model_name = 'prednet-L_all-mul-peepFalse-tbiasFalse-best'
model_file = os.path.join(MODEL_DIR, model_name + '.pt')




reading_files = sys.argv[1]
customize_tag = reading_files[:-4]
with Image.open(f"img/{reading_files}") as img:
	img = img.convert('RGB')  
	
	img = np.asarray(img)
	print(f"img {img.shape}")
	img = np.transpose(img, (2, 0, 1))
	input_video = torch.from_numpy(img)
	input_video = input_video.unsqueeze(0).repeat(nt, 1, 1, 1).unsqueeze(0)
	
	print(f"input {input_video.shape}")

input_size = input_video.shape[3:5] #(128, 160)
print(f"input_size {input_size}")


model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
				extrap_start_time=extrap_start_time, peephole=peephole, lstm_tied_bias=lstm_tied_bias)
model.load_state_dict(torch.load(model_file))

print('Model: ' + model_name)
if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()


make_video = True
make_plot = True
if make_video:
	pred = model(input_video / 255.).detach()
	pred = pred.cpu()
	pred = pred.permute(0,4,2,3,1) # [1, 10, 3, 128, 160]) 
	pred = torch.squeeze(pred, 0)
	pred = pred.numpy() 
	print(f"pred {pred.shape}") # (10, 160, 128, 3)


	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	import matplotlib.animation as animation

	img = pred # some array of images
	frames = [] # for storing the generated images
	fig = plt.figure()
	print(f"correct size {img[0].shape} {input_video.shape} --now {(input_video / 255.)[0, 0].numpy().transpose(1, 2, 0).shape}")
	frames.append([plt.imshow((input_video / 255.)[0, 0].numpy().transpose(1, 2, 0), animated=True)])
	for i in range(1, 2):
		frames.append([plt.imshow((img[i] * 255).astype(np.uint8), animated=True)])

	ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
									repeat_delay=10000)
	
	os.makedirs('snake_video/', exist_ok=True)
	ani.save(f'snake_video/snake_pred-{model_name}-extrap_start_time{extrap_start_time}-nt-{nt}-{customize_tag}.mp4')
	plt.show()

if make_plot:
	pred = model(input_video / 255.) # (batch_size, channels, width, height, nt)
	pred = pred.cpu()
	pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)

	step = 0
	targets = input_video
	targets = targets.detach().numpy() 
	# targets = targets / targets.max()
	pred = pred.detach().numpy()  
	# pred = pred / pred.max()

	targets = np.transpose(targets, (0, 1, 3, 4, 2)) 
	print(f"targets {targets.shape}")
	# targets = targets / targets.max(1, keepdims=True)
	pred = np.transpose(pred, (0, 1, 3, 4, 2)) 

	print(f"pred {pred.shape}")
	# pred = pred / pred.max(1, keepdims=True)

	targets = targets.astype(int)
	pred = pred * 255.
	pred = pred.astype(int)

	aspect_ratio = float(pred.shape[2]) / pred.shape[3]
	fig = plt.figure(figsize = (nt, 2*aspect_ratio))
	gs = gridspec.GridSpec(2, nt)
	gs.update(wspace=0., hspace=0.)
	os.makedirs("snake_plots", exist_ok=True)
	plot_save_dir = os.path.join('snake_plots/') # NOTE: change path
	if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
	plot_idx = np.random.permutation(targets.shape[0])[:n_plot]
	
	for i in plot_idx:
		for t in range(nt):
			plt.subplot(gs[t])  
			plt.imshow( targets[i,t], interpolation='nearest', cmap=plt.get_cmap('gray')) # requires 'channel_last'
			plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
			if t==0: plt.ylabel('Actual', fontsize=10)

			plt.subplot(gs[t + nt])
			plt.imshow( pred[i,t], interpolation='nearest',  cmap='gray')
			plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
			if t==0: plt.ylabel('Predicted', fontsize=10)

			# add optical flow calculation on here using opencv
		
		img_filename = f"snake_results-{model_name}-extrap_start_time{extrap_start_time}-nt-{nt}-{customize_tag}"
		print('Saving ' + img_filename)
		img_filename = img_filename
		plt.savefig(img_filename + '.png')
		plt.clf()
		print('Image Saved')


	for t in range(3):
		fig = plt.figure(figsize = (5, 5))
		plt.imshow(pred[0,t])
		plt.savefig(os.path.join('snake_plots/', img_filename+f'-{t}-{customize_tag}.png'))
		plt.clf()
	
	
	fig = plt.figure(figsize = (5, 5))
	plt.imshow(targets[0,0])
	plt.savefig(os.path.join('snake_plots/', img_filename+f'-target-{customize_tag}.png'))
	plt.clf()

