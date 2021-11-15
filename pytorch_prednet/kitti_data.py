import hickle as hkl
import numpy as np
import torch
import torch.utils.data as data



class KITTI(data.Dataset):
	def __init__(self, datafile, sourcefile, nt, output_mode='error', sequence_start_mode='all',
				N_seq=None, shuffle=False, data_format='channels_first'):
		"""
		Arguments
		---------
		datafile: str
			path to data file
		sourcefile: str
			path to source file
		nt: int
			number of frames in a sequence
		output_mode: str
			['error', prediction']
		sequence_start_mode: str
			['all', 'unique']
		N_seq: int
			size of the subset of sequences uses
		shuffle: boolean
			toggle shuffle before taking subset
		data_format: str
			['channels_first', 'channels_last']
		"""

		self.datafile = datafile
		self.sourcefile = sourcefile
		self.X = hkl.load(self.datafile) # dims: (num_images, num_cols, num_rows, num_channels)
		self.sources = hkl.load(self.sourcefile) # source for each image: ensures consec. frames are from same video
		self.nt = nt
		self.data_format = data_format
		assert sequence_start_mode in {'all', 'unique'}, 'sequences_start_mode must be in {all, unique}'
		self.sequence_start_mode = sequence_start_mode
		assert output_mode in {'error', 'prediction'}, 'output_mode must b in {errror, prediction}'			
		self.output_mode = output_mode

		if self.data_format == 'channels_first':
			self.X = np.transpose(self.X, (0,3,1,2)) # NOTE: Image shape is now (N,C,W,H)
		self.im_shape = self.X[0].shape

		cur_loc = 0
		possible_starts = []
		if self.sequence_start_mode == 'all':
			# allows for all possible sequence when all nt frames are from same source
			possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) 
										if self.sources[i] == self.sources[i + self.nt - 1]])
		elif self.sequence_start_mode == 'unique':
			# creates sequences where each unique frame is in at most one sequence
			while cur_loc < self.X.shape[0] - self.nt + 1:
				if self.sources[cur_loc] == self.sources[cur_loc + self.nt - 1]: 
					possible_starts.append(cur_loc)
					cur_loc += self.nt
				else:
					cur_loc += 1
		self.possible_starts = possible_starts

		if shuffle:
			self.possible_starts = np.random.permutation(self.possible_starts)
		if N_seq is not None and len(self.possible_starts) > N_seq: # select subset of sequences
			self.possible_starts = self.possible_starts[:N_seq]
		self.N_sequences = len(self.possible_starts)

	def preprocess(self, X):
		return X.astype(np.float32) / 255.
    
	def __getitem__(self, index):
		loc = self.possible_starts[index]
		X = self.preprocess(self.X[loc:loc+self.nt])
		if self.output_mode == 'error':
			y = np.zeros((1,), np.float32)
		elif self.output_mode == 'prediction':
			y = X
		return (X, y)

	def __len__(self):
		return self.N_sequences


