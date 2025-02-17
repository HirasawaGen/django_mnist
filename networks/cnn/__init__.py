import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T

from networks.cnn.model import CNN
from networks.cnn.train import train

if __name__ == '__main__':
	model = CNN()
	model.load_state_dict(torch.load('./weights/cnn.pth'))
	train_dataset = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())

