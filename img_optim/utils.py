"""
Assorted utilities.
"""
import torch
import scipy
from scipy import ndimage
import numpy as np
import skimage

# TODO: make all these work on either GPU or CPU

IMGNET_MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).transpose(1, 3)
IMGNET_STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).transpose(1, 3)

# there's probably some better way to do this
C_IMGNET_MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).transpose(1, 3).cuda()
C_IMGNET_STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).transpose(1, 3).cuda()

def normalize_img(img):
    """
    Prepare a 0-255 image (on CPU) for giving to a torchvision model.
    """
    # bring to 0-1 range, then normalize with mean+std from imagenet
    return ((img.float() / 255.0) - IMGNET_MEAN) / IMGNET_STD

def clip_img(img):
    """
    Clip a torchvision-scaled image (on GPU) between 0 and 255.
    """
    transformed = (img * C_IMGNET_STD) + C_IMGNET_MEAN
    transformed[transformed < 0] = 0
    transformed[transformed > 1] = 1

    return (transformed - C_IMGNET_MEAN) / C_IMGNET_STD

def torch_gaussian_filter(img, sigma):
	"""
	Smooth a GPU torch tensor with a gaussian filter.
	"""
	np_img = img.detach().cpu().numpy()
	# print(img.shape)
	# TODO: experiment smoothing over time dimension (with a diff sigma value)
	return torch.tensor(ndimage.gaussian_filter(np_img, sigma=(0, 0, 0, sigma, sigma)), requires_grad=True).cuda()

def torch_rand_range(shape, min_max):
    """
    Randomly initialize a tensor in a particular range
    """
    return (torch.rand(*shape) * (abs(min_max[1] - min_max[0]))) + min_max[0]

def get_atomic_layers(network):
    """
    Gets all the single layers (e.g. Conv2d, MaxPool2d) of a
    net; in order of operation if it's a standard feedforward
    network.
    """
    children = list(network.modules())
    atomic_kids = [c for c in children if len(list(c.children())) == 0]

    return atomic_kids

def add_noise(tensor, stddev):
    """
    Adds normal noise with the given standard deviation to the
    tensor.
    """
    dist = torch.distributions.normal.Normal(loc=0, scale=stddev)

    return tensor + dist.sample(tensor.shape).cuda()

def torch_check_grayscale(img):
    """
    Check if a torchvision image (BCHW) is grayscale (meaning
    all channels are equal).
    """
    if img.shape[1] == 1:
        return True

    return torch.allclose(img[:,0,:,:], img[:,1,:,:], atol=1e-6) and \
            torch.allclose(img[:,1,:,:], img[:,2,:,:], atol=1e-6)

def load_img_torchvision(fname):
    """
    Load an image from disk, into a format torchvision models
    can work with. 

    Returns:
    
    the B3HW image (on CPU)

    a flag that's true if the image was grayscale (it has 3
    channels regardless)
    """
    img = skimage.io.imread(fname)
    grayscale = False

    if len(img.shape) < 3 or img.shape[2] == 1:
        # it's grayscale, fix that
        img = skimage.color.gray2rgb(img)
        grayscale = True

    # crop to center 224
    h,w = img.shape[:2]
    img = img[int((h - 224) / 2) : int(224 + (h - 224) / 2),
            int((w - 224) / 2) : int(224 + (w - 224) / 2), :]

    # put into BCHW order
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    return normalize_img(torch.tensor(img)), grayscale
