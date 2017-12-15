import torch
import torch.nn.functional
import numpy as np
from torch import nn
import math
from torch.autograd import Variable


def squash(vector, axis=-1):
	vec_np = vector.data.numpy()
	vec_norm = np.sum(vec_np ** 2, axis=2)
	vec_norm = vec_norm[:, :, np.newaxis]
	coeff = vec_norm / ((1 + vec_norm) * np.sqrt(vec_norm))
	return Variable(torch.from_numpy(coeff)) * vector

class PrimaryCapsules(nn.Module):
	def __init__(self, channels_in, channels_out, capsules_dim, kernels, stride):
		super(PrimaryCapsules, self).__init__()
		temp = []
		for i in range(capsules_dim):
			temp.append(nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=kernels, stride=stride))
		self.primaryCapsules = nn.ModuleList(temp)

	def forward(self, x):
		temp = []
		for capsule in self.primaryCapsules:
			temp.append(capsule(x).view(x.size(0), -1, 1))
		ret = torch.cat(temp, dim=-1)

		ret = squash(ret)
		return ret
class DigitCapsules(nn.Module):
	def __init__(self, channels_in, channels_out, capsules_dim, num_routing, num_capsules):
		super(DigitCapsules, self).__init__()
		self.capsules_dim = capsules_dim
		self.num_routing = num_routing
		self.num_capsules = num_capsules
		self.dc_w = nn.Parameter(torch.randn(capsules_dim, num_capsules, channels_in, channels_out))
	def forward(self, x):
		self.u = x[np.newaxis, :, :, np.newaxis, :].clone() @ self.dc_w[:, np.newaxis, :, :, :].clone()
		self.b = Variable(torch.zeros(*self.u.size()))
		for i in range(self.num_routing):
			sm = nn.Softmax(dim=2)
			self.c = sm(self.b)
			self.s = (self.c * self.u).sum(dim=2, keepdim=True)
			self.v = squash(self.s)
			if i != self.num_routing - 1:
				self.b = self.b + (self.u * self.v).sum(dim=-1, keepdim=True)
		return self.v





