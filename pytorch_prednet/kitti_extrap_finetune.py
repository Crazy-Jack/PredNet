import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from kitti_data import KITTI
from prednet import PredNet

from debug import info

# Define losss as MAE of frame prediction after t=0
# It's invalid to compute loss on error representation, since the error isn't calc'd w.r.t ground truth
def extrap_loss(y_true, y_hat):
	y_true = y_true[:, 1:]
	y_hat  = y_hat[:, 1:]
	# 0.5 to match scale of loass hwen trained in error mode
	return 0.5 * torch.mean(torch.abs(y_true - y_hat)) 

# Training parameters
num_epochs = 150
batch_size = 4 # 16
lr = 0.001 # if epoch < 75 else 0.0001
nt = 15 # num of time steps
n_train_seq = 500
n_val_seq = 125 # 80-20 "split"
extrap_start_time = 10

# Model parameters
loss_mode = 'L_0'
peephole = False
lstm_tied_bias = False
gating_mode = 'mul'

A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

if loss_mode == 'L_0':
	layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
elif loss_mode == 'L_all':
	layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.1], [0.1], [0.1]]).cuda())

time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1) # lambda_t's in Lotter et al. 2017
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

# Directories
DATA_DIR = '/user_data/hrockwel/pytorch-prednet/kitti_data/'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')

val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Location of t+1 model
MODEL_DIR = './models/'
base_model_name = 'prednet-L_0-mul-peepFalse-tbiasFalse'
base_model_file = os.path.join(MODEL_DIR, base_model_name + '.pt')

kitti_train = KITTI(train_file, train_sources, nt, output_mode='prediction', N_seq=n_train_seq)
kitti_val = KITTI(val_file, val_sources, nt, output_mode='prediction',  N_seq=n_val_seq)
input_size = kitti_train.im_shape[1:3] #(128, 160)
num_train_steps = len(kitti_train)//batch_size

train_loader = DataLoader(kitti_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(kitti_val, batch_size=batch_size, shuffle=True)

model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
				extrap_start_time=extrap_start_time, peephole=peephole, lstm_tied_bias=lstm_tied_bias)
model.load_state_dict(torch.load(base_model_file))

if torch.cuda.is_available():
	print('Using GPU.')
	model.cuda()

print('Old Model: prednet-{}-{}-peep{}-tbias{}'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)) 
print('New Model: prednet-{}-{}-peep{}-tbias{}-finetuned'.format(loss_mode, gating_mode, peephole, lstm_tied_bias)) 

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
		inputs = Variable(inputs.cuda()) 
		targets = Variable(targets.cuda())
		pred  = model(inputs)
		pred = pred.permute(0,4,1,2,3)
		loss = extrap_loss(targets, pred)
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
			pred = model(inputs)
			pred = pred.permute(0,4,1,2,3)
			loss = extrap_loss(targets, pred)
			val_loss += loss.item() * batch_size

	val_loss /= len(kitti_val)
	print('Validation loss: {:.6f}'.format(val_loss))
	if val_loss < min_val_loss:
		print('Validation Loss Decreased: {:.6f} --> {:.6f} \t Saving the Model'.format(min_val_loss, val_loss))
		min_val_loss = val_loss
		# Save model
		torch.save(model.state_dict(), 'prednet-best-{}-{}-peep{}-tbias{}-finetuned.pt'.format(loss_mode, gating_mode, peephole, lstm_tied_bias))
	print

# Save model
torch.save(model.state_dict(), 'prednet-{}-{}-peep{}-tbias{}-finetuned.pt'.format(loss_mode, gating_mode, peephole, lstm_tied_bias))
