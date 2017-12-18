import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import Model_foolbox as Model
#from torchvision.datasets.mnist import MNIST
from torch import nn
# import torchnet as tnt
import foolbox
import torch.utils.data


from mnist import MNIST
mndata = MNIST('.\python-mnist-0.3\data')
images, labels = mndata.load_training()
# print(len(images[0]))
# print(labels.shape)
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
# print(type(images))

model = Model.Net()
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 255), num_classes=10, cuda=False, channel_axis=1, preprocessing=(0, 1))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# print(images)
# dataset = MNIST(root='./data', download=True, train=True)
# dataset_test = MNIST(root='./data', download=True, train=False)
# data = getattr(dataset, 'train_data')
# labels = getattr(dataset, 'train_labels')
# tensor_dataset = tnt.dataset.TensorDataset([data, labels])

# print(data[:, np.newaxis, :, :].float().numpy().shape)
# adversarial = attack(data[:, :, :].float().numpy(), labels.float().numpy())
# tensor_dataset.parallel(batch_size=batch_size, num_workers=4, shuffle=True)
tensor_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(images).view(-1, 28, 28), torch.LongTensor(labels))
train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
train_loader_foolbox = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                           batch_size=1, 
                                           shuffle=True)
# data_test = getattr(dataset_test, 'test_data')
# labels_test = getattr(dataset_test, 'test_labels')
# print(data_test.shape)
# print(tensor_dataset.shape)

# Loss = Model.forward()
criterion = foolbox.criteria.TargetClass(2)
attack = foolbox.attacks.LBFGSAttack(fmodel, criterion)
for i, (images, labels) in enumerate(train_loader_foolbox):

	images = Variable(torch.FloatTensor(images.numpy()))
	labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)
	labels = Variable(labels)
	# outputs = fmodel(images)
	adversarial = attack(images.float().data.numpy(), labels.long().data.numpy())



for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# print(images)
		images = Variable(torch.FloatTensor(images[:, np.newaxis :, :].numpy()))
		# labels = Variable(labels)
		# print(images)
		# print(labels)
		# labels = torch.LongTensor(labels)
		labels = torch.sparse.torch.eye(10).index_select(dim=0, index=labels)
		labels = Variable(labels)
		optimizer.zero_grad()
        
        # Forward + Backward + Optimize
		# print(len(images.float().numpy()))
		# adversarial = attack(images.float().numpy(), labels.float().numpy())
		outputs = model(images)
		loss = model.loss(labels.clone(), outputs, images.clone())
		# loss = (labels, outputs)
		

		loss.backward()

		optimizer.step()

		if (i+1) % 10 == 1:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
				%(epoch+1, num_epochs, i+1, len(tensor_dataset)//batch_size, loss.data[0]))
			if (i+1) % 80 == 0:
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
