from __future__ import print_function
import torch

"""
Assorted loss functions for use in optimization.
All take in at least three arguments: the real output, 
the target output, and the image (for regularization), 
but don't necessary use all of these.

Partial application can be used to change the default
arguments beyond those three.

Most default arguments,  come from the paper
'Understanding Deep Image Representations by Inverting Them'.
"""

# Global Constants
# TODO
IMG_AXES = (1,2,3)
CHAN_AXIS = 1
SPATIAL_AXES = (2, 3)
SRF_SIZE = {'A1':4, 'A2':10, 'A3':22,'E0':1, 'E1':4, 'E2':10, 'E3':22, 'R0':3, 'R1':8, 'R2':18, 'R3':38}
MAX_NAT_OUTPUT={'R0': 0.9216051, 'R1': 0.9999999, 'R2': 0.99999994, 'R3': 1.0, 'A1': 2.931553,\
				'A3': 14.336471, 'A2': 6.3082156, 'E1': 2.4927447, 'E0': 1.0, 'E3': 14.336471, 'E2': 6.3082156}

def alpha_norm(output, target, image, alpha=6):
    """
    Takes the alpha-norm of the mean-subtracted
    image, but with the mean, not the sum, to better
    account for variable-size images without changing
    hyperparameters.
    """
    return torch.norm(image - image.mean(), alpha)


def tv_norm(image, beta=2):
    """
    Takes the total variation norm of the image.
    """
    col_shift = torch.empty(image.shape, requires_grad=False).cuda()
    row_shift = torch.empty(image.shape, requires_grad=False).cuda()

	# TODO: Generalize
    row_shift[:, :, :, 1:, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :]) ** 2
    col_shift[:, :, :, :, 1:] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1]) ** 2

    return torch.norm(row_shift + col_shift, beta / 2)

def tv(image, beta=2):
    """
    Takes the total variation of the image.
    """
    col_shift = torch.empty(image.shape, requires_grad=False).cuda()
    row_shift = torch.empty(image.shape, requires_grad=False).cuda()

	# TODO: Generalize
    row_shift[:, :, :, 1:, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :]) ** 2
    col_shift[:, :, :, :, 1:] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1]) ** 2

    return torch.pow(row_shift + col_shift, beta / 2).sum(axis=IMG_AXES)

def output_loss(output, target=None, idx=None):
	"""
	Euclidean distance between the output and target
	at a specific slice of the tensors.
	"""
	if idx is None:
		idx = (slice(None), ) * self.shape.numel()

	mask = torch.zeros_like(output)
	mask[idx] = 1.0

	if target is None:
		loss = torch.linalg.norm(output * mask, axis=IMG_AXES)
	else:
		loss =  torch.linalg.norm((output - target) * mask, axis=IMG_AXES)
	return loss

def L2_reg_loss(output, image, unit, idx, lam, maximize=True, resp_type='sustained'):
	"""
	Karen Simonyan, Andrea Vedaldi, Andrew Zisserman 2014
	"""
	# Constants
	nt = output.size(-1)
	
	# Drop-out neurons over which we don't want to optimize
	mask = torch.zeros_like(output)
	mask[idx] = 1.0
	output *= mask
	if unit == 'E':
		output = torch.abs(output) # directionality of error is not needed	
	
	# Average error units (within a time step)
	loss = output.sum(axis=IMG_AXES) / mask.sum(axis=IMG_AXES)
	# print(loss.data)
	
	# Flip sigs depending on max/min
	if maximize:
		loss *= -1
	else:
		lam *= -1
	
	# Final Energy formulation
	loss = loss + lam * torch.linalg.norm(image)
	
	# Weighted sum of error time-components
	loss = time_loss_weight(loss, nt, resp_type)
	
	return loss

def natural_preimage_loss(output, image, vis_type, unit_layer=None, idx=None, target=None, maximize=None):
	"""
	Aravindh Mahendran, Andrea Vedaldi 2016
	"""
	# Constants
	C = 1
	alpha = 6
	beta = 2
	B = 80
	B_p = 2 * B
	V = B/6.5
	H, W = image.size(SPATIAL_AXES)
	nt = output.size(-1)

	# Computing main loss term
	if vis_type == 'inversion':
		if idx == None:
			idx = (slice(None),) * len(output.shape)
		# Target is image_0
		assert target is not None, 'target must be a Tensor'
		maximize = True
		loss = output_loss(output, target, idx)/output_loss(target,idx)

	elif vis_type == 'activation maximimization':
		if idx == None:
			idx = (slice(None),) * len(output.shape)
		assert maximize is not None, 'maximize must be a boolean'
		# Drop-out neurons over which we don't want to optimize
		mask = torch.zeros_like(output)
		mask[idx] = 1.0
		output *= mask 
		Z = SRF_SIZE[unit_layer] * MAX_NAT_OUTPUT[unit_layer]
		loss  = output.sum(axis=IMG_AXES) / Z 
	
	elif vis_type == 'caricaturization':
		assert target is not None, 'target must be a Tensor > 0'
		assert maximize is not None, 'maximize must be a boolean'
		# Target is ReLU(CNN(image_0))
		output *= target
		Z = torch.linalg.norm(target, axis=IMG_AXES)
		loss = output.sum(axis=IMG_AXES) / Z 

	# Computing regularizers
	R_TV =  tv(image, beta)/(H * W * V**beta)	
	rgb_iso =  torch.pow( torch.linalg.norm(image, axis=CHAN_AXIS, keepdims=True), alpha/2 ).sum(axis=IMG_AXES)
	N_alpha = rgb_iso.sum(axis=IMG_AXES) / (H * W * B**alpha) # soft constraint
	if torch.all(torch.pow(rgb_iso, .5) <= B_p): # hard constraint
		R_alpha = N_alpha
	else:
		R_alpha = float('inf')
	
	# Flip signs depending on max/min
	if maximize: 
		loss *= -1	
	else:
		R_TV *= -1
		R_alpha *= -1
	
	# Energy formulation
	loss = loss + R_TV + R_alpha # (batch, nt)
	
	# Weighted sum over time-components
	loss = time_loss_weight(loss, nt)

	return loss

def time_loss_weight(loss, nt, resp_type):
	"""
	Lotter et al. 2017
	"""
	if nt > 1:
		assert resp_type in ['sustained', 'impulse'], 'Invalid arguments'
		if resp_type == 'sustained':
			time_loss_weights = 1./(nt - 1) * torch.ones((nt, 1)) 
			time_loss_weights[0] = 0
			time_loss_weights.requires_grad_(True)
		elif resp_type == 'impulse':
			time_loss_weights = torch.zeros((nt, 1))
			time_loss_weights[-1] = 1.
			time_loss_weights.requires_grad_(True)
	else:
		time_loss_weights = torch.ones((nt, 1), requires_grad=True) # lambda_t's in Lotter et al. 2017
	time_loss_weights = time_loss_weights.cuda()
	
	loss = torch.mm(loss.view(-1, nt), time_loss_weights) 
	
	return loss


def maximization_loss(output, target, image, idx):
	"""
	Loss intended to maximize a particular output index.
	"""

	return -output[idx]

