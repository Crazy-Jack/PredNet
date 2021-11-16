from __future__ import print_function
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kitti_data import KITTI
from kitti_settings import *
from prednet import PredNet

def init_weights(m):
	""""
	init_weights:
	Initialize sub-module weights using xavier_uniform (for kernels) and 0. (for kernel biases)

	Arguments:
	_________
	m: nn.Module
	"""

	if isinstance(m, nn.Conv2d):
		# Reset kernels and biases
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

# Training parameters
num_epochs = 150
batch_size = 4 # 16
lr = 0.001 # if epoch < 75 else 0.0001
nt = 10 # num of time steps
n_train_seq = 500
n_val_seq = 100

# Model parameters
loss_mode = 'L_all'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'

default_channels = (3, 48, 96, 192)
channel_six_layers = (3, 48, 96, 192, 384, 768)
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)
using_default_channels = A_channels == default_channels
num_layers = len(A_channels)

if loss_mode == 'L_0':
	layer_loss_weights = torch.zeros((num_layers, 1), device='cuda')
	layer_loss_weights[0,0] = 1.
elif loss_mode == 'L_all':
	layer_loss_weights = 0.1 * torch.ones((num_layers, 1), device='cuda')
	layer_loss_weights[0] = 1.

time_loss_weights = 1./(nt - 1) * torch.ones((nt, 1), device='cuda') # lambda_t's in Lotter et al. 2017
time_loss_weights[0] = 0

# Directories

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')

val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

kitti_train = KITTI(train_file, train_sources, nt, output_mode='error', N_seq=n_train_seq)
kitti_val = KITTI(val_file, val_sources, nt, output_mode='error',  N_seq=n_val_seq)
input_size = kitti_train.im_shape[1:3] #(128, 160)
num_train_steps = len(kitti_train)//batch_size

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True)

model = PredNet(input_size, R_channels, A_channels, output_mode='error', gating_mode=gating_mode,
				peephole=peephole, lstm_tied_bias=lstm_tied_bias)

if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()
model.apply(init_weights)

if using_default_channels:
	model_name = 'prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)
else:
	channels_str = '_'.join([str(x) for x in A_channels])
	model_name = 'prednet-{}-{}-peep{}-tbias{}-chans_{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias, channels_str)

print('Model: ' + model_name)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07)
criterion = nn.L1Loss()

def lr_scheduler(optimizer, epoch):
	if epoch < num_epochs //2:
		return optimizer
	else:
		for param_group in optimizer.param_groups:
			param_group['lr'] = 0.0001
		return optimizer

min_val_loss = float('inf')



for epoch in range(num_epochs):

	# ----------------- Training Loop ----------------------
	train_loss = 0.0
	optimizer = lr_scheduler(optimizer, epoch)
	model.train()

	for step, (inputs, targets) in enumerate(train_loader):
		# batch x time_steps x channel x width x height
		inputs = inputs.cuda()
		targets = targets.cuda()
		# Refer to Eqn (5) in Lotter et al. 2017
		# L_train = Sum_t( lam_t * Sum_l( lam_l/nl * Sum_{n_l}(E^t_l) ) )
		errors = model(inputs) # batch x n_layers x nt
		loc_batch = errors.size(0)
		# Weighted sum of error time-components
		# (batch*n_layers x nt)(nt x 1) -->  batch*n_layers x 1
		loss = torch.mm(errors.view(-1, nt), time_loss_weights) 	
		# Weighted sum of error layer-components
		# (batch x n_layer)(n_layer x 1) --> batch x 1 
		loss = torch.mm(loss.view(loc_batch, -1), layer_loss_weights) 	
		# Average batch los
		# train_loss = torch.mean(train_loss, dim=0, keepdim=True)

		# Calculate Mean Absolute Error
		loss = criterion(loss, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * batch_size

		if step % 10 == 0:
			print('step: {}/{}, loss: {:.6f}'.format(step, num_train_steps, loss))

	train_loss /= len(kitti_train)
	print('Epoch: {}/{}, loss: {:.6f}'.format(epoch+1, num_epochs, train_loss)) 
    
	# ------------------  Validation Loop  -------------------
	model.eval()
	val_loss = 0.0
	with torch.no_grad():
		for step, (inputs, targets) in enumerate(val_loader):
			# batch x time_steps x channels x width x heigth
			inputs = inputs.cuda()
			targets = targets.cuda()
			errors = model(inputs) # barch x n_layers x nt
			loc_batch = errors.size(0)
			# Weighted sum of error time-components
			loss = torch.mm(errors.view(-1, nt), time_loss_weights)
			# Weighted sum of error layer-components
			loss = torch.mm(loss.view(loc_batch, -1), layer_loss_weights)
			# Calculate Mean Absolute Error
			loss = criterion(loss, targets)
			val_loss += loss.item() * batch_size

	val_loss /= len(kitti_val)
	print('Validation loss: {:.6f}'.format(val_loss))
	if val_loss < min_val_loss:
		print('Validation Loss Decreased: {:.6f} --> {:.6f} \t Saving the Model'.format(min_val_loss, val_loss))
		min_val_loss = val_loss
		# Save model
		torch.save(model.state_dict(), model_name + '-best.pt')
	print()

# Save model
torch.save(model.state_dict(), model_name + '.pt')
