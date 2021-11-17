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

# Visualization parameters
n_plot = 4 # number of plot to make (must be <= batch_size)
DATA_DIR = 'kitti_raw_data'


# Model parameters
gating_mode = 'mul'
peephole = False
lstm_tied_bias = False
nt = 10  # TODO: change the number of time
extrap_start_time = None  # TODO: change to None if don't want to do extrapolation
batch_size = 4

default_channels = (3, 48, 96, 192)
channel_six_layers = (3, 48, 96, 192, 384, 768)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

MODEL_DIR = 'models/'
model_name = 'prednet-L_0-mul-peepFalse-tbiasFalse-best'  # TODO: Change the model to other models
model_file = os.path.join(MODEL_DIR, model_name + '.pt')

RESULTS_SAVE_DIR = './'

kitti_test = KITTI(test_file, test_sources, nt, output_mode='prediction', sequence_start_mode='all')
num_steps = len(kitti_test)//batch_size
test_loader = DataLoader(kitti_test, batch_size=batch_size, shuffle=False)

input_size = kitti_test.im_shape[1:3] #(128, 160)

model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
				extrap_start_time=extrap_start_time, peephole=peephole, lstm_tied_bias=lstm_tied_bias)
model.load_state_dict(torch.load(model_file))

print('Model: ' + model_name)
if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()

pred_MSE = 0.0
copy_last_MSE = 0.0
for step, (inputs, targets) in enumerate(test_loader):
	# ---------------------------- Test Loop -----------------------------
	# print(f"inputs {inputs.shape}")
	inputs = inputs.cuda() # batch x time_steps x channel x width x height
	
	targets = targets

	pred = model(inputs) # (batch_size, channels, width, height, nt)
	pred = pred.cpu()
	pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)

	if step == 0:	
		print('inputs: ', inputs.size())
		print('targets: ', targets.size())
		print('predicted: ', pred.size(), pred.dtype)

	pred_MSE += torch.mean((targets[:, 1:] - pred[:, 1:])**2).item() # look at all timesteps after the first
	copy_last_MSE += torch.mean((targets[:, 1:] - targets[:, :-1])**2).item()
	
	if step == 20: # change this number to control the starting sequence of your video clip
		# Plot some predictions
		targets = targets.detach().numpy() * 255.
		pred = pred.detach().numpy() * 255.

		targets = np.transpose(targets, (0, 1, 3, 4, 2))
		pred = np.transpose(pred, (0, 1, 3, 4, 2))
		
		targets = targets.astype(int)
		pred = pred.astype(int)

		aspect_ratio = float(pred.shape[2]) / pred.shape[3]
		fig = plt.figure(figsize = (nt, 2*aspect_ratio))
		gs = gridspec.GridSpec(2, nt)
		gs.update(wspace=0., hspace=0.)
		plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/') # NOTE: change path
		if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
		plot_idx = np.random.permutation(targets.shape[0])[:n_plot]
		for i in plot_idx:
			for t in range(nt):
				plt.subplot(gs[t])
				plt.imshow(targets[i,t], interpolation='none') # requires 'channel_last'
				plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
				if t==0: plt.ylabel('Actual', fontsize=10)

				plt.subplot(gs[t + nt])
				plt.imshow(pred[i,t], interpolation='none')
				plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
				if t==0: plt.ylabel('Predicted', fontsize=10)
			
			img_filename = model_name + '-step_' + str(step)
			if extrap_start_time is not None:
				img_filename = img_filename + '-extrap_' + str(extrap_start_time) + '+' + str(nt - extrap_start_time)
			img_filename = img_filename + '-plot_' + str(i)

			while os.path.exists(img_filename + '.png'):
				img_filename += '_'
			
			print('Saving ' + img_filename)
			img_filename = plot_save_dir + img_filename
			plt.savefig(img_filename + '.png')
			plt.clf()
			print(f'Image Saved as {img_filename}.png')
		
		break

# Calculate dataset MSE
pred_MSE /= num_steps
copy_last_MSE /= num_steps

print('Prediction MSE: {:.6f}'.format(pred_MSE)) # no need to worry about "first time step"
print('Copy-Last MSE: {:.6f}'.format(copy_last_MSE))

