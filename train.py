import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import Model
from torchvision.datasets.mnist import MNIST
from torch import nn
import torchnet as tnt
import foolbox

# from torch.autograd import Variable
#     from torch.optim import Adam
#     from torchnet.engine import Engine
#     from torchnet.logger import VisdomPlotLogger, VisdomLogger
#     from torchvision.utils import make_grid
#     from torchvision.datasets.mnist import MNIST
#     from tqdm import tqdm
#     import torchnet as tnt

num_epochs = 1
batch_size = 100
learning_rate = 0.005


# BATCH_SIZE = 100
# NUM_CLASSES = 10
# NUM_EPOCHS = 500
# NUM_ROUTING_ITERATIONS = 3


model = Model.Net()
# fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 255), num_classes=10, cuda=False, channel_axis=1, preprocessing=(0, 1))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = MNIST(root='./data', download=True, train=True)
dataset_test = MNIST(root='./data', download=True, train=False)
data = getattr(dataset, 'train_data')
labels = getattr(dataset, 'train_labels')
tensor_dataset = tnt.dataset.TensorDataset([data, labels])
# attack = foolbox.attacks.FGSM(fmodel)
# print(data[:, np.newaxis, :, :].float().numpy().shape)
# adversarial = attack(data[:, :, :].float().numpy(), labels.float().numpy())
# tensor_dataset.parallel(batch_size=batch_size, num_workers=4, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
data_test = getattr(dataset_test, 'test_data')
labels_test = getattr(dataset_test, 'test_labels')
# print(data_test.shape)
# print(tensor_dataset.shape)

# Loss = Model.forward()


for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# print(images)
		images = Variable(torch.FloatTensor(images[:, np.newaxis :, :].numpy()))
		# labels = Variable(labels)
		# print(images)
		labels = torch.LongTensor(labels)
		labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)
		labels = Variable(labels)
		optimizer.zero_grad()
        
        # Forward + Backward + Optimize
		# print(len(images.float().numpy()))
		# adversarial = attack(images.float().numpy(), labels.float().numpy())
		outputs, reconstruction = model(images)
		loss = model.loss(labels.clone(), outputs, images.clone(), reconstruction.clone())
		# loss = (labels, outputs)
		

		loss.backward()

		optimizer.step()

		if (i+1) % 10 == 1:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
				%(epoch+1, num_epochs, i+1, len(tensor_dataset)//batch_size, loss))
			if (i+1) % 80 == 1:
				images = Variable(torch.FloatTensor(data_test[:200, np.newaxis, :, :].numpy()))
				# print(labels_test[:200].shape)
				outputs, reconstruction = model(images)
				# outputs
				_, max_length_indices = outputs.max(dim=1)
				labels_true = labels_test[:200].numpy()
				labels_pre = max_length_indices.data.numpy()
				# print(labels_true - labels_pre)
				# print(np.count_nonzero(labels_pre - labels_true))
				acc = 1 - (np.count_nonzero(labels_true - labels_pre) * 1.0 / len(labels_pre))


				# print(max_length_indices.size())
				print("accuracy: " + str(acc))
