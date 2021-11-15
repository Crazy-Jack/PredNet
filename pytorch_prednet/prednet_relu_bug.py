import torch
import torch.nn as nn
from torch.nn import functional as F
from conv_lstm_cell import ConvLSTMCell
from torch.autograd import Variable

class PredNet(nn.Module):
	def __init__(self, input_size, R_channels, A_channels, p_max=1.0, output_mode='error',
				gating_mode='mul', extrap_start_time=None, peephole=True, lstm_tied_bias=True):
		"""
		Arguments
		---------

		input_size: (int, int)
			dimensions of frames (required for peephole connections)
		R_channels:
			number of channel for RNN in each layer
		A_channels:
			number of channels in A_l in each layer
		p_max: float
			Maximum pixel value in input images
		output_mode: str
			Controls what is outputted by Prednet.
			Either 'error', 'prediction', 'all', or layer specification.
			If 'error' the mean response of the error (E) units of each layer will be outputted.
				That is, the output shape will be (batch_size, n_layers, nt).
			If 'prediction', the frame prediction will be outputted.
			If 'pred+err' the output will be the frame predicition concatenated with the mean layer-errors over time
				The frame prediction is flattened before concatenation
			If '<unit_type>+<layer_num>, the features of a particular layer will be outputted
				Unit types are 'R', 'Ahat', 'A', and 'E'
				Ex: 'Ahat2' is the prediction generated at the third layer
		gating_mode: str
			Controls the gating operation for the ConvLSTM cells
			Either 'mul' or 'sub' (multiplicative vs. subtractive)
		extrap_start_time: int
			Frame to begin using past predictions as ground-truth
		peephole: boolean
			To include/exclude peephole connections in ConvLSTM
		lstm_tied_bias: boolean
			To use tied/untied bias in ConvLSTM convolutions
		"""
		
		super(PredNet, self).__init__()
		self.r_channels = R_channels + (0, )  # for convenience
		self.a_channels = A_channels
		self.n_layers = len(R_channels)
		self.input_size = input_size
		self.output_mode = output_mode
		self.gating_mode = gating_mode
		self.extrap_start_time = extrap_start_time
		self.peephole = peephole
		self.lstm_tied_bias = lstm_tied_bias
		self.p_max = p_max
		
		# Input validity checks
		default_output_modes = ['prediction', 'error', 'pred+err']
		layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
		default_gating_modes = ['mul', 'sub']
		assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
		assert gating_mode in default_gating_modes, 'Invalid gating_mode: ' + str(gating_mode)
		
		if self.output_mode in layer_output_modes:
			self.output_layer_type = self.output_mode[:-1]
			self.output_layer_num = int(self.output_mode[-1])
		else:
			self.output_layer_type = None
			self.output_layer_num = None

		h, w = self.input_size

		for i in range(self.n_layers):
			# A_channels multiplied by 2 because E_l concactenates pred-target and target-pred
			# Hidden states don't have same size due to upsampling
			# How does this handle i = L-1 (final layer) | appends a zero

			if self.gating_mode == 'mul':	
				cell = ConvLSTMCell((h, w), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
									(3, 3), gating_mode='mul', peephole=self.peephole, tied_bias=self.lstm_tied_bias)
			elif self.gating_mode == 'sub':
				cell = ConvLSTMCell((h, w), 2 * self.a_channels[i] + self.r_channels[i+1], self.r_channels[i],
									(3, 3), gating_mode='sub', peephole=self.peephole, tied_bias=self.lstm_tied_bias)

			setattr(self, 'cell{}'.format(i), cell)
			h = h // 2
			w = w // 2

		for i in range(self.n_layers):
			# Calculate predictions A_hat
			conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
			setattr(self, 'conv{}'.format(i), conv)

		self.upsample = nn.Upsample(scale_factor=2)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		for l in range(self.n_layers - 1):
			# Propagate error as next layer's target (line 16 of Lotter algo)
			# In channels = 2 * A_channels[l] because of pos/neg error concat
			# NOTE: Operation belongs to curr layer l and produces next layer  state l+1

			update_A = nn.Sequential(nn.Conv2d(2* self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
			setattr(self, 'update_A{}'.format(l), update_A)
	
	def set_output_mode(self, output_mode):
		"""
		set_output_mode:
			Resets output mode
		Arguments
		_________
		output_mode: str
		"""
		
		# Change output mode
		self.output_mode = output_mode

		# Input validity checks
		default_output_modes = ['prediction', 'error', 'pred+err']
		layer_output_modes = [unit + str(l) for l in range(self.n_layers) for unit in ['R', 'E', 'A', 'Ahat']]
		assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
		
		if self.output_mode in layer_output_modes:
			self.output_layer_type = self.output_mode[:-1]
			self.output_layer_num = int(self.output_mode[-1])
		else:
			self.output_layer_type = None
			self.output_layer_num = None
	
	def step(self, a, states):
		"""
		step:
		Performs inference for a single time step

		Arguments:
		_________
		a: Tensor
			target image frame
		states: list
			contains layer states -->  [R + C + E]
			if self.extrap_start_time --> [R + C + E, prev_pred, t]
		"""
		
		batch_size = a.size(0)
		R_layers = states[:self.n_layers]
		C_layers = states[self.n_layers:2*self.n_layers]
		E_layers = states[2*self.n_layers:3*self.n_layers]

		if self.extrap_start_time is not None:
			t = states[-1]
			if t >= self.extrap_start_time: # if past self.extra_start_time use previous prediction as input
				a = states[-2]

		# Update representation units
		for l in reversed(range(self.n_layers)):
			cell = getattr(self, 'cell{}'.format(l))
			r_tm1 = R_layers[l]
			c_tm1 = C_layers[l]
			e_tm1 = E_layers[l]
			if l == self.n_layers - 1:
				r, c = cell(e_tm1, (r_tm1, c_tm1))
			else:
				tmp = torch.cat((e_tm1, self.upsample(R_layers[l+1])), 1)
				r, c = cell(tmp, (r_tm1, c_tm1))
			R_layers[l] = r
			C_layers[l] = c

		# Perform error forward pass
		for l in range(self.n_layers):
			conv = getattr(self, 'conv{}'.format(l))
			a_hat = conv(R_layers[l])
			if l == 0:
				a_hat= torch.min(a_hat, torch.tensor(self.p_max).cuda()) # alternative SatLU (Lotter)
				frame_prediction = a_hat
			pos = F.relu(a_hat - a)
			neg = F.relu(a - a_hat)
			e = torch.cat([pos, neg],1)
			E_layers[l] = e
			
			# Handling layer-specific outputs
			if self.output_layer_num == l:
				if self.output_layer_type == 'A':
					output = a
				elif self.output_layer_type == 'Ahat':
					output = a_hat
				elif self.output_layer_type == 'R':
					output = R_layers[l]
				elif self.output_layer_type == 'E':
					output = E_layers[l]

			if l < self.n_layers - 1: # updating A for next layer
				update_A = getattr(self, 'update_A{}'.format(l))
				a = update_A(e)

		if self.output_layer_type is None:
			if self.output_mode == 'prediction':
				output = frame_prediction
			else:
				# Batch flatten (return 2D matrix) then mean over units
				# Finally, concatenate layers (batch, n_layers)
				mean_E_layers = torch.cat([torch.mean(e.view(batch_size, -1), axis=1, keepdim=True) for e in E_layers], axis=1)
				if self.output_mode == 'error':
					output = mean_E_layers
				else:
					output = torch.cat([frame_prediction.view(batch_size, -1), mean_E_layers], axis=1)

		states = R_layers + C_layers + E_layers
		if self.extrap_start_time is not None:
			states += [frame_prediction, t+1]
		return output, states

	def forward(self, input):
		"""
		forward:

		Perform inference on a sequence of frames

		Arguments:
		input: Tensor
			A (batch_size, nt, num_channels, height, width) tensor
		"""

		R_layers = [None] * self.n_layers
		C_layers = [None] * self.n_layers
		E_layers = [None] * self.n_layers

		h, w = self.input_size # input.size(-2), input.size(-1)
		batch_size = input.size(0)

		# Initialize states
		for l in range(self.n_layers):
			R_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True).cuda()
			C_layers[l] = torch.zeros(batch_size, self.r_channels[l], h, w, requires_grad=True).cuda()
			E_layers[l] = torch.zeros(batch_size, 2*self.a_channels[l], h, w, requires_grad=True).cuda()
			# Size of hidden state halves from each layer to the next
			h = h//2
			w = w//2

		states = R_layers + C_layers + E_layers
		# Initialize previous_prediction
		if self.extrap_start_time is not None:
			frame_prediction = torch.zeros_like(input[:,0], dtype=torch.float32).cuda()
			states += [frame_prediction, -1] # [a, t]
			
		num_time_steps = input.size(1)
		total_output = [] # contains output sequence
		for t in range(num_time_steps):
			a = input[:,t].type(torch.cuda.FloatTensor)
			output, states = self.step(a, states)
			total_output.append(output)

		ax = len(output.shape)
		# print(output.shape)
		total_output = [out.view(out.shape + (1,)) for out in total_output]
		total_output = torch.cat(total_output, axis=ax) # (batch, ..., nt)
		return total_output
		
