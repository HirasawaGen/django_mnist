from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST

import torchvision.transforms as T


class LimitedDataset(Dataset):
	def __init__(self, indeces, dataset):
		self.indexes = indeces
		self.dataset = dataset

	def __len__(self):
		return len(self.indexes)

	def __getitem__(self, index):
		return self.dataset[self.indexes[index]]


def get_seperated_mnist_datasets():
	train_dataset = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
	indexes_list = [[] for _ in range(10)]
	for i, (image, label) in enumerate(train_dataset):
		indexes_list[label].append(i)
	return tuple(LimitedDataset(indexes_list[i], train_dataset) for i in range(10))
