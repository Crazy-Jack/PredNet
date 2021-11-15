from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from prednet import PredNet
from losses import *
from utils import *
from functools import partial
from nat_preimage_optimizer import Optimizer

# Directories
RESULTS_SAVE_DIR = '/user_data/aumoren/uPNC2021-aniekan/img_optim/'

# Location of t+1 model
MODEL_DIR = '../pytorch_prednet/models/'
model_name = 'prednet-L_0-mul-peepFalse-tbiasFalse'
model_file = os.path.join(MODEL_DIR, model_name + '.pt')

# Training parameters
nt = 5 # num of time steps
max_iter = 500

# Model parameters
loss_mode = 'L_0'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'
extrap_start_time = None
img_shape = (128, 160)
unit_layer = 'A3'
maximize = True

# NOTE: Ahat cannot be used to optimize as gradients do not flow to images
# A, E, H, and C units can be used
"""
Unit type sizes (for each layer)

A3: (1, 192, 16, 20, nt) 
E3: (1, 384, 16, 20, nt)

A2: (1, 96, 32, 40, nt)
E2: (1, 192, 32, 40, nt)

A1: (1, 48, 64, 80, nt)
E1: (1, 96, 64, 80, nt)

A0: (1, 3, 128, 160, nt)
E0: (1, 6, 128, 160, nt) 

"""
# L3
# idx =  (0,slice(0,192),7,9,slice(None)) 
idx =  (0,0,7,9,slice(None))
# idx =  (0,slice(None),7,9,slice(None))

# L2
# idx =  (0,slice(0,192),7,9,slice(None)) 
# idx =  (0,0,16,20,slice(None))
# idx =  (0,slice(None),16,20,slice(None))

# L1
# idx =  (0,slice(0,192),7,9,slice(None)) 
# idx =  (0,0,32,40,slice(None))
# idx =  (0,slice(None),32,40,slice(None))

# L0
# idx =  (0,slice(0,192),7,9,slice(None)) 
# idx =  (0,0,64,80,slice(None))
# idx =  (0,slice(None),7,9,slice(None))

layer = None

def plot_prediction(targets, pred):
	"""
	plot_predictions:

	Plot a row of ground-true, and a row of predictions

	Arguments
	________

	n_plot (int <= batch_size):
		Visualization param controlling number of figures to create
	"""	
	
	aspect_ratio = float(pred.shape[2]) / pred.shape[3]
	fig = plt.figure(figsize = (nt, 2*aspect_ratio))
	gs = gridspec.GridSpec(2, nt)
	gs.update(wspace=0., hspace=0.)
	plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'plots/')

	if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
	
	for t in range(nt):
		plt.subplot(gs[t])
		plt.imshow(targets[0,t], interpolation='none') # requires 'channel_last'
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Actual', fontsize=10)

		plt.subplot(gs[t + nt])
		plt.imshow(pred[0,t], interpolation='none')
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Predicted', fontsize=10)
	
	idx_str = ''
	for i in idx:
		if type(i) is int:
			idx_str = idx_str + '_' + str(i)
		else:
			idx_str = idx_str + '_' + 'sN' 

	img_filename = plot_save_dir +  model_name + '-n_iter_' + str(max_iter) + '-nt_' + str(nt) + \
					'-unit_' + unit_layer + '-idx' + idx_str + '-max_' + str(maximize) + '-lam_' + str(lam) + '-sigma_' + str(sigma)
	if extrap_start_time is not None:
		img_filename = img_filename + '-extrap_' + str(extrap_start_time) + '+' + str(nt - extrap_start_time)

	while os.path.exists(img_filename + '.png'):
		img_filename += '_'
	
	print('Saving ' + img_filename + '.png')
	plt.savefig(img_filename + '.png')
	plt.clf()
	print('Image Saved')

if __name__ == '__main__':
	
	# Loading PredNet model
	A_channels = (3, 48, 96, 192)
	R_channels = (3, 48, 96, 192)
	model = PredNet(img_shape, R_channels, A_channels, output_mode=unit_layer,
					gating_mode=gating_mode, peephole=peephole, lstm_tied_bias=lstm_tied_bias)
	model.load_state_dict(torch.load(model_file))

	if torch.cuda.is_available():
		print('Using GPU.')
		model.cuda()

	print('Old Model: prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)) 
	layers = get_atomic_layers(model) # hal's "layers" are actually modules
	input_layer = layers[42]
	
	# partial application, since the index can't be passed in optimizer code
	# loss_func = partial(maximization_loss, idx=idx)
	loss_func = partial(natural_preimage_loss, unit_layer=unit_layer, idx=idx, lam=lam, maximize=maximize)
	optimizer = Optimizer(network, layer, loss_func, input_layer)

	# Depends on visualization type
	# TODO
	target = None

	# NOTE: Optimizer will incorrectly pick the input_layer to be Cell0
	# image dims: (batch, nt, channels, height, width)
	rand_img = torch_rand_range((1, nt, 3,) + img_shape, (0, 1)).cuda()
	optim_img, loss =  optimizer.optimize(rand_img, target, max_iter=max_iter,
										lr=lr, sigma=sigma, debug=True)
	# Save optimized image
	with torch.no_grad():
		# TODO: check ig unit activity
		model.output_mode = 'prediction'
		model.output_layer_type = None

		pred = model(optim_img)	
		pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)
		
		optim_img = optim_img.cpu()
		pred = pred.cpu()
		
	# Normalizing images between [0, 1]
	optim_img = optim_img.detach().numpy() * 255.
	pred = pred.detach().numpy() * 255.
	
	optim_img = np.transpose(optim_img, (0, 1, 3, 4, 2))
	pred = np.transpose(pred, (0, 1, 3, 4, 2))
	
	optim_img = optim_img.astype(int)
	pred = pred.astype(int)
	
	plot_prediction(optim_img, pred)

