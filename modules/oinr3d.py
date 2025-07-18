'''
File describing O-INR model for 3D volume
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class INR(nn.Module):
	def __init__(self, inchannel, outchannel,
                 hidden_layers=4, out_features=1, 
                 outermost_linear=True, first_omega_0=0,
                 hidden_omega_0=0, scale=1, pos_encode=False,
                 sidelength=1, fn_samples=None, use_nyquist=None):
		super(INR, self).__init__()

		self.inchannel = inchannel
		self.outchannel = out_features
		print("model inchannel:", inchannel)
		print("model outchannel:", out_features)

		self.approx_conv1 = nn.Conv3d(inchannel, 64, 3, 1, padding='same')
		self.approx_conv2 = nn.Conv3d(64, 128, 3, 1, padding='same')
		self.approx_conv3 = nn.Conv3d(128, 128, 3, 1, padding='same')
		self.approx_conv4 = nn.Conv3d(128, 64, 3, 1, padding='same')
		self.approx_conv5 = nn.Conv3d(64, self.outchannel, 3, 1, padding='same')

	def forward(self, x, verbose=False, autocast=False):
		device = "cuda" if x.is_cuda else "cpu"
		with torch.autocast(device, enabled=autocast):
			return self._forward(x)

	def _forward(self, x):
		x = self.approx_conv1(x)
		x = torch.sin(x)
		x = self.approx_conv2(x)
		x = torch.sin(x)
		x = self.approx_conv3(x)
		x = torch.sin(x)
		x = self.approx_conv4(x)
		x = torch.sin(x)
		x = self.approx_conv5(x)
		return x