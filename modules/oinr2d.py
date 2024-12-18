'''
Model defined in this file cater to O-INR
for 2d image
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_sequence(start, end):
    result = []
    current = start
    
    while current < end:
        result.append([current, current * 2])
        current *= 2
    
    result.append([end, end])
    
    current = end
    while current > start:
        result.append([current, current // 2])
        current //= 2
        
    return result

class INR(nn.Module):
	def __init__(self, in_features, hidden_features,
                 hidden_layers=4, out_features=1, 
                 outermost_linear=True, first_omega_0=0,
                 hidden_omega_0=0, scale=1, pos_encode=False,
                 sidelength=1, fn_samples=None, use_nyquist=None):
		super(INR, self).__init__()

		self.inchannel = in_features
		self.outchannel = out_features

		self.start_features = 64

		self.scale_list = [[self.inchannel, self.start_features]]
		self.scale_list.extend(generate_sequence(self.start_features, hidden_features))
		self.scale_list.append([self.start_features, self.outchannel])

		self.convs = nn.ModuleList(
			[nn.Conv2d(in_f, out_f, 3, 1, padding='same', bias=False)
    			for in_f, out_f in self.scale_list]
        )
		# print(self.scale_list)
		# [[27, 64], [64, 128], [128, 256], [256, 512], [512, 512],
		#  [512, 256], [256, 128], [128, 64], [64, 1]]             
	
	def forward(self, x, verbose=False):
		x = x.permute(0, 2, 1).unsqueeze(-1)

		# Apply each convolution layer with sinusoidal activation
		for i, conv in enumerate(self.convs):
			x = conv(x)
			if i < len(self.convs) - 1:  # Don't apply activation after last layer
				x = torch.sin(x)
			
			if verbose:
				print(f"Layer {i} output shape: {x.shape}")

		signal = x
		return signal, None