from __future__ import print_function
import os

import torch
import torch.nn as nn
import numpy as np
import random

# from prednet import PredNet # NOTE: uncomment me
from prednet_relu_bug import PredNet
from utils import get_atomic_layers
from losses import L2_reg_loss
from optimizer import Optimizer
from utils import torch_rand_range
from plot_utils import create_movie_frames
from functools import partial

from wrappers import visualize

# Setting seed
# random.seed(123)

# Directories
RESULTS_SAVE_DIR = '/user_data/aumoren/uPNC2021-aniekan/img_optim/'
movie_save_dir = os.path.join(RESULTS_SAVE_DIR, 'movies_relu_bug/') # NOTE: change path

if not os.path.exists(movie_save_dir): os.mkdir(movie_save_dir)

# Training parameters
max_iter = 500
lr_max = 1
lr_min = .01
lr = np.linspace(lr_max, lr_min, max_iter) # schedule

lam = 0
sigma_max = 3
sigma_min = .5
sigma = np.linspace(sigma_max, sigma_min, max_iter)
grayscale = False
min_loss_val = None

nt = 10 # num of time steps
maximize = True
resp_type = 'impulse'

if min_loss_val is not None:
	early_stopper = lambda losses: losses[-1] < min_loss_val
else:
	early_stopper = None

train_params = {}
train_params['lr'] = lr
train_params['lr_min'] = lr_min
train_params['lr_max'] = lr_max
train_params['sigma'] = sigma
train_params['sigma_max'] = sigma_max
train_params['sigma_min'] = sigma_min
train_params['grayscale'] = grayscale
train_params['nt'] = nt
train_params['maximize'] = maximize
train_params['resp_type'] = resp_type

# Model parameters
loss_mode = 'L_0'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'
extrap_start_time = None

# Location of t+1 model
# MODEL_DIR = '../pytorch_prednet/models/' # NOTE: uncomment me
MODEL_DIR = '../pytorch_prednet/archive/models_preReLU/'
model_name = 'prednet-L_0-mul-peepFalse-tbiasFalse'
model_file = os.path.join(MODEL_DIR, model_name + '.pt')
img_shape = (128, 160)

# Probe information
# Unit type sizes (for each layer)
UNIT_LAYER_SHAPE = {}
UNIT_LAYER_SHAPE['A3'] = (1, 192, 16, 20, nt) 
UNIT_LAYER_SHAPE['E3'] = (1, 384, 16, 20, nt)
UNIT_LAYER_SHAPE['R3'] = (1, 192, 16, 20, nt)

UNIT_LAYER_SHAPE['A2'] = (1, 96, 32, 40, nt)
UNIT_LAYER_SHAPE['E2'] = (1, 192, 32, 40, nt)
UNIT_LAYER_SHAPE['R2'] = (1, 96, 32, 40, nt)

UNIT_LAYER_SHAPE['A1'] = (1, 48, 64, 80, nt)
UNIT_LAYER_SHAPE['E1'] = (1, 96, 64, 80, nt)
UNIT_LAYER_SHAPE['R1'] = (1, 48, 64, 80, nt)

UNIT_LAYER_SHAPE['A0'] = (1, 3, 128, 160, nt)
UNIT_LAYER_SHAPE['E0'] = (1, 6, 128, 160, nt) 
UNIT_LAYER_SHAPE['R0'] = (1, 3, 128, 160, nt)


if __name__ == '__main__':
	# Loading PredNet model
	A_channels = (3, 48, 96, 192)
	R_channels = (3, 48, 96, 192)
	model = PredNet(img_shape, R_channels, A_channels, output_mode='prediction',
					gating_mode=gating_mode, peephole=peephole, lstm_tied_bias=lstm_tied_bias)
	model.load_state_dict(torch.load(model_file))

	if torch.cuda.is_available():
		print('Using GPU.')
		model.cuda()

	print('Model: prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)) 
	network_modules = get_atomic_layers(model) # hal's "layers" are actually modules
	
	# image dims: (batch, nt, channels, height, width)
	target = None
	rand_img = torch_rand_range((nt,3) + img_shape, (0,1)).unsqueeze(0).cuda()
	for unit_layer in ['R3','A3']:
		indices = []
		unit_layer_shape = UNIT_LAYER_SHAPE[unit_layer]
		# rows = (unit_layer_shape[2]//2,)
		# cols = (unit_layer_shape[3]//2,)
		# rows = random.sample( range(unit_layer_shape[2]//4, 3 * unit_layer_shape[2]//4), 4)
		# cols = random.sample( range(unit_layer_shape[3]//4, 3 * unit_layer_shape[3]//4), 4)
		rows = [11] #[4, 6, 11, 7]
		cols = [12] #[9, 6, 5, 8]
		channels = [136] #(random.randint(0, unit_layer_shape[1]),)
		print(rows)
		print(cols)
		for row in rows:
			for col in cols :
				# channels = (random.randint(0, unit_layer_shape[1]),)
				# channels = (slice(None),)
				# channels = (random.randint(0, unit_layer_shape[1]) for i in range(5))
				# channels = (i  for i in range(unit_layer_shape[1]))
				for channel in channels:
					indices.append((0,channel,row,col,slice(None)))
		
		for idx in indices:
			model.set_output_mode(unit_layer)
			'''	
			optim_img, rand_img, loss = visualize(model, None, unit_layer[0], idx, img_shape=(nt,3,) + img_shape, init_range=(0, 1),
						max_iter=max_iter, lr=lr, lam=lam,  maximize=maximize, resp_type=resp_type, sigma=sigma, input_layer=None, grayscale=grayscale, debug=True)
			'''
			
			# partial application, since the index can't be passed in optimizer code
			loss_func = partial(L2_reg_loss, unit=unit_layer[0], idx=idx, lam=lam, maximize=maximize, resp_type=resp_type)
			optimizer = Optimizer(model, None, loss_func)

			# now start optimization
			optim_img, loss = optimizer.optimize(rand_img.detach().clone(), target, max_iter=max_iter,
					lr=lr, sigma=sigma, grayscale=grayscale, debug=False, early_stopper=early_stopper)
			

			# Save optimized image
			with torch.no_grad():
				# Check that activity increased
				rand_output = model(rand_img)
				print('Random output: ', rand_output[idx])
				optim_output = model(optim_img)
				print('Optimized output: ', optim_output[idx])
				
				model.set_output_mode('prediction')

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
			
			create_movie_frames(optim_img, pred, model_name, unit_layer, idx, train_params, movie_save_dir)

