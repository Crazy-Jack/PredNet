from __future__ import print_function
import torch
import numpy as np
from utils import *

class Optimizer():
	"""
	Optimize an image to produce some result in a deep net.
	"""

	def __init__(self, net, layer, loss_func, input_layer=None):
		"""
		Parameters:

		net: nn.Module 
			presumably a deep net
		layer: nn.Module or None
			part of the network that gives relevant output
			if None will use network output
		input_layer: nn.Module
			the input layer of the network; will try
			to determine automatically if not specified
		loss_func: callable 
			taking layer output, target output, and image,
			returning the loss
		"""

		self.net = net
		self.net.requires_grad_(True)
		self.layer = layer
		self.loss_func = loss_func
		self.input_layer = input_layer

		# will only define hooks during optimization so they can be removed
		self.acts = []

	def optimize(self, image, target, constant_area=None, max_iter=1000,
			lr=np.linspace(5, 0.5, 1000), clip_image=False,
			grayscale=False, sigma=0, debug=False, early_stopper=None):
		"""
		Parameters:

		image: image to start from, presumably where the target was 
		modified from

		target: target activation, to be passed into loss_func

		constant_area: indices such that image[0:1, 2:3, :] stays
		constant each iteration of gradient ascent
		
		max_iter: maximum number of iterations to run

		lr: 'learning rate' (multiplier of gradient added to image at
		each step, or iterable of same length as max_iter with varying values)

		clip_image: whether or not to clip the image to real (0-256) pixel
		values, with standard torchvision transformations

		sigma: sigma of the Gaussian smoothing at each iteration
		(default value 0 means no smoothing), can be an iterable of
		length max_iter like 'lr'

		debug: whether or not to print loss each iteration

		early_stopper: function that takes the list of losses so far,
		returns True if the optimization process should stop

		Returns:

		optimized image
		loss for the last iteration
		"""

		image.requires_grad_(False)
		new_img = image.detach().clone().requires_grad_(True)
		
		# change it to an array even if it's constant, for the iterating code
		if isinstance(lr, int) or isinstance(lr, float):
			lr = [lr] * max_iter

		if isinstance(sigma, float) or isinstance(sigma, int):
			sigma = [sigma] * max_iter

		# want the actual, atomic first layer object for hooking
		if self.input_layer is None:
			children = [child for child in self.net.modules()
					if len(list(child.children())) == 0]
			input_layer = children[0]
		else:
			input_layer = self.input_layer

		print('input_layer: ', input_layer)
		
		# Setting up hooks
		if self.layer is not None:
			forw_hook = self.layer.register_forward_hook(
					lambda m,i,o: self.acts.append(o)) # self.acts.append(i) to optimize on pre-activations
		
		# now do gradient ascent
		losses = []
		print('In Optimizer')
		for i in range(max_iter):
			# get gradient
			out = self.net(new_img)
			# print('out shape: ', out.shape)

			if self.layer is None:
				self.acts.append(out)	

			loss = self.loss_func(self.acts[0], new_img)
			losses.append(loss)
			self.net.zero_grad()
			loss.backward()
			
			if debug:
				print('loss for iter {}: {}'.format(i, loss.item()))

			# all processing of gradient was done in loss_func
			# even momentum if applicable; none is done here
			with torch.no_grad():
				# Utilize gradient that flowed back to image
				# print('new_img grad: ', new_img.grad)
				new_img.data = new_img - lr[i] * new_img.grad
				
				# Clip image between [0,1]
				p_min = new_img.min()
				p_max = new_img.max()
				new_img.data = (new_img - p_min)/(p_max - p_min)

				if clip_image:
					new_img.data = clip_img(new_img)

				if sigma[i] > 0:
					new_img.data = torch_gaussian_filter(new_img, sigma[i])

				if constant_area is not None:
					# assuming torchvision structure (BCHW) here
					# TODO: change this
					new_img.data[:, :, constant_area[0]:constant_area[1],
							constant_area[2]:constant_area[3]] = image[:, :,
									constant_area[0]:constant_area[1],
									constant_area[2]:constant_area[3]]

				if grayscale:
					# keep the image grayscale
					gray_vals = new_img.mean(1)
					new_img.data = torch.stack([gray_vals] * 3, 1)
			
			# Reset 
			del self.acts[:]

			if early_stopper is not None and early_stopper(losses):
				print('early stopping at iter {}'.format(i))
				break

		# avoid side effects
		if self.layer is not None:
			forw_hook.remove()

		return new_img, loss

