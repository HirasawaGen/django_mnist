from torch import nn
import torchvision.transforms as T


transform = T.Compose([
	T.Normalize((0.5,), (0.5,))
])


de_transform = T.Compose([  # 将单通道图片中[-1, 1]范围的图片转换为[0, 1]范围的图片
	T.Normalize((-1,), (2,))
])


class Generator(nn.Module):
	def __init__(self, latent_dim=1000, img_shape=(1, 28, 28)):
		super().__init__()
		self.latent_dim = latent_dim
		self.img_shape = img_shape
		self.model = nn.Sequential(
			nn.Linear(latent_dim, 256),
			nn.LeakyReLU(0.2),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2),
			nn.Linear(1024, int(img_shape[0] * img_shape[1] * img_shape[2])),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.model(x)
		x = x.view(-1, *self.img_shape)
		if not self.training:
			x = de_transform(x)
		return x


class Discriminator(nn.Module):
	def __init__(self, img_shape=(1, 28, 28)):
		super().__init__()
		self.img_dim = int(img_shape[1] * img_shape[2])
		self.img_channels = img_shape[0]
		self.img_shape = img_shape
		self.model = nn.Sequential(
			nn.Linear(self.img_dim, 512),
			nn.LeakyReLU(0.2),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		if not self.training:
			x = transform(x)
		batch_size = x.size(0)
		x = x.view(batch_size, -1)
		return self.model(x)
