import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.features = nn.Sequential(
			# [batch_size, 1, 28, 28]
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			# [batch_size, 32, 28, 28]
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# [batch_size, 32, 14, 14]
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			# [batch_size, 64, 14, 14]
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			# [batch_size, 64, 7, 7]
		)
		self.classifier = nn.Sequential(
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 10),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		x = F.log_softmax(x, dim=1)
		return x
