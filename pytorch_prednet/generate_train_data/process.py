'''
Code for processing a data set of video frames
'''
import os
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl


desired_im_sz = (128, 160)
DATA_DIR = './data/'
stim_type = 'radial_large'


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():

    im_list = []
    im_dir = DATA_DIR #os.path.join(DATA_DIR) 
    files = list(os.walk(im_dir, topdown=True))[-1][-1] # get all frames in DATA_DIR

    im_list += [im_dir + f for f in sorted(files)]

    print( 'Creating ' + stim_type + ' data: ' + str(len(im_list)) + ' images')
    X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
    for i, im_file in enumerate(im_list):
        im = imread(im_file)
        X[i] = process_im(im, desired_im_sz)

    hkl.dump(X, os.path.join('./', 'X_' + stim_type + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    process_data()
