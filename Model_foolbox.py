import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
import capsuleLayer




class MarginLoss(nn.Module):
	def __init__(self):
		super(MarginLoss, self).__init__()
	
	# def forward(self, labels, classes):
	# 	left = F.relu(0.9 - classes, inplace=True) ** 2
	# 	right = F.relu(classes - 0.1, inplace=True) ** 2

	# 	margin_loss = labels * left + 0.5 * (1. - labels) * right
	# 	margin_loss = margin_loss.sum()

	# 	# reconstruction_loss = self.reconstruction_loss(reconstructions, images)
	# 	print("forward ")
	# 	print(margin_loss / 100)

	# 	return (margin_loss) / 100
	# def backward(self):
	# 	super(MarginLoss, self).backward(self.loss)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
		self.capsuleLayer1 = capsuleLayer.PrimaryCapsules(channels_in=256, channels_out=32, capsules_dim=8, kernels=9, stride=2)
		self.capsuleLayer2 = capsuleLayer.DigitCapsules(channels_in=8, channels_out=16, capsules_dim=10, num_routing=3, num_capsules=32 * 6 * 6)
		self.fc1 = nn.Linear(16 * 10, 512)
		self.fc2 = nn.Linear(512, 1024)
		self.fc3 = nn.Linear(1024, 784)

	def forward(self, x):
		print("hererererere")
		print(x.size())
		x1 = F.relu(self.conv1(x.clone().view(x.size(0), 1, 28, 28)))
		# print(x.shape)
		# print(x.clone().view(100, 1, 28, 28).shape)
		x1 = self.capsuleLayer1(x1)
		x1 = self.capsuleLayer2(x1)
		x1 = x1.squeeze()
		print(x1.size())
		if x1.size() == (10, 16):
			print("size equal here")
			x1 = x1[:, np.newaxis, :]
		x1 = x1.transpose(0, 1)
		x1 = x1.transpose(1, 2)
		# print(x)


		
		norms = (x1 ** 2).sum(dim=1) ** 0.5
		norms = F.softmax(norms)
		# print(norms.shape)
		_, max_length_indices = norms.max(dim=1)
		y = Variable(torch.sparse.torch.eye(10)).index_select(dim=0, index=Variable(max_length_indices.data))
		# print(x1.transpose(1, 2).shape)
		y = F.relu(self.fc1((x1.transpose(1, 2) * y[:, :, np.newaxis].clone()).view(x1.size(0), -1)))
		y = F.relu(self.fc2(y))
		y = F.softmax(self.fc3(y))
		self.reconstructions = y
		# print(norms.shape == (100, 10))
		print(norms.size())
		return norms
	def loss(self, y_true, y_pred, images):
		loss = (y_true.clone() * (F.relu(0.9 - y_pred.clone()) ** 2) + 0.5 * (1 - y_true.clone()) * (F.relu(y_pred.clone() - 0.1) ** 2)).sum()
		# print("forward: ")
		# print(loss)
		# print(images.size())
		# print(reconstructions.size())
		return loss / 100 + torch.mean(torch.abs(images - self.reconstructions.view(100, 1, 28, 28))) * 0.0005 / 100