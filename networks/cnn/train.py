import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
import torch.nn.functional as F

from tqdm import tqdm
from networks.cnn.model import CNN


def train(
		model=None,
		device=None,
		save_path=None,
		batch_size=4,
		threshold=0.95,
		max_epochs=200
):
	if model is None:
		model = CNN()
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if save_path is None:
		save_path = './weights/cnn.pth'
	model.to(device)
	train_dataset = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_dataset = MNIST(root='./data', train=False, download=True, transform=T.ToTensor())
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	epochs = 0
	max_accuracy = 0.0
	accuracy = 0.0
	while epochs < max_epochs:
		for images, labels in tqdm(
				train_loader,
				desc=f'Epoch {epochs+1} Training, Current ACC: {accuracy:.4f}, Max ACC: {max_accuracy:.4f}'
		):
			labels = F.one_hot(labels, num_classes=10).float()
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
		correct = 0
		total = 0
		with torch.no_grad():
			for images, labels in tqdm(
					test_loader,
					desc=f'Epoch {epochs+1} Testing, Current ACC: {accuracy:.4f}, Max ACC: {max_accuracy:.4f}'
			):
				images = images.to(device)
				labels = labels.to(device).float()
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		accuracy = correct / total
		if accuracy > max_accuracy:
			torch.save(model.state_dict(), save_path)
			max_accuracy = accuracy
		if accuracy > threshold:
			break
		epochs += 1
	return model
