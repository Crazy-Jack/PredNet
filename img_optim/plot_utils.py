from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
import glob

import os
import numpy as np

def plot_prediction(targets, pred, unit_layer, idx, save_dir):
	"""
	plot_predictions:

	Plot a row of ground-true, and a row of predictions

	Arguments
	________

	n_plot (int <= batch_size):
		Visualization param controlling number of figures to create
	"""		
	nt = targets.shape[1]
	aspect_ratio = float(pred.shape[2]) / pred.shape[3]
	fig = plt.figure(figsize = (nt, 2*aspect_ratio))
	gs = gridspec.GridSpec(2, nt)
	gs.update(wspace=0., hspace=0.)
	
	for t in range(nt):
		plt.subplot(gs[t])
		plt.imshow(targets[0,t], interpolation='none') # requires 'channel_last'
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Actual', fontsize=10)

		plt.subplot(gs[t + nt])
		plt.imshow(pred[0,t], interpolation='none')
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		if t==0: plt.ylabel('Predicted', fontsize=10)
	
	# Create filename
	img_filename = 'static_frames'
	print(save_dir)
	print('Saving ' + img_filename + '.png')
	plt.savefig(save_dir + img_filename + '.png')
	plt.close('all')
	print('Image Saved')



def create_movie_frames(targets, pred, model_name, unit_layer, idx, train_params, movie_save_dir):
	"""
	create_movie_frame: 
		Creates ordered frames that can be used to make a gif
		./movies/
			movie_name/
				targets/
					frame*.png
				pred/
					frame*.png
	Arguments
	_________
	TODO
	"""
	
	# Unpack training (hyper) parameters
	nt = train_params['nt']
	sigma = train_params['sigma']
	sigma_max = train_params['sigma_max']
	sigma_min = train_params['sigma_min']
	lr = train_params['lr']
	lr_max = train_params['lr_max']
	lr_min = train_params['lr_min']
	grayscale = train_params['grayscale']
	maximize = train_params['maximize']

	# Useful strings
	idx_str = ''
	for i in idx:
		if type(i) is int:
			idx_str = idx_str + '_' + str(i)
		else:
			idx_str = idx_str + '_' + 'sN' 
	if type(sigma) is not int:
		sigma_str = str(sigma_max) + '_' + str(sigma_min)
	else:
		sigma_str = str(sigma)
	if type(lr) is not int:
		lr_str = str(lr_max) + '_' + str(lr_min)
	else:
		lr_str = str(lr)

	movie_name =  model_name  + '-nt_' + str(nt) + '-unit_' + unit_layer +\
					'-idx' + idx_str + '-max_' + str(maximize) +\
					'-gray_' + str(grayscale)
	
	movie_save_subdir = os.path.join(movie_save_dir, movie_name + '/')
	if not os.path.exists(movie_save_subdir): os.mkdir(movie_save_subdir)

	targets_save_dir = os.path.join(movie_save_subdir, 'targets/')
	if not os.path.exists(targets_save_dir): os.mkdir(targets_save_dir)
	
	pred_save_dir = os.path.join(movie_save_subdir, 'pred/')
	if not os.path.exists(pred_save_dir): os.mkdir(pred_save_dir)
	
	# Plotting static frames
	plot_prediction(targets, pred, unit_layer, idx, movie_save_subdir)
	
	for t in range(nt):
		t_str = str(t)
		if t < 10:
			t_str = '0' + t_str
		
		# Plotting optimized targets
		plt.figure()
		plt.imshow(targets[0,t], interpolation='none') # requires 'channel_last'
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		
		img_filename = 'frame_' + t_str
		img_save_dir = targets_save_dir + img_filename

		print('Saving ' + img_filename + '.png')
		plt.savefig(img_save_dir + '.png')
		plt.clf()
		print('Image Saved')
			
		# Plotting predictions
		plt.figure()
		plt.imshow(pred[0,t], interpolation='none') # requires 'channel_last'
		plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
		
		img_filename = 'frame_' + t_str
		img_save_dir = pred_save_dir + img_filename

		print('Saving ' + img_filename + '.png')
		plt.savefig(img_save_dir + '.png')
		plt.clf()
		print('Image Saved')
	
	# Create animated gifs
	targets_frames = []
	targets_imgs = glob.glob(targets_save_dir + '*.png')
	for i in targets_imgs:
		new_frame = Image.open(i)
		targets_frames.append(new_frame)
	# Save into a GIF file that loops forever
	targets_frames[0].save(targets_save_dir + 'movie.gif', format='GIF', append_images=targets_frames[1:],\
							quality=95, save_all=True, duration=200, loop=0)

	pred_frames = []
	pred_imgs = glob.glob(pred_save_dir + '*.png')
	for i in pred_imgs:
		new_frame = Image.open(i)
		pred_frames.append(new_frame)
	# Save into a GIF file that loops forever
	pred_frames[0].save(pred_save_dir + 'movie.gif', format='GIF', append_images=pred_frames[1:], save_all=True, duration=100, loop=0)
	plt.close('all')
