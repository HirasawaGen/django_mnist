import torch
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from networks.gan.models import Generator, Discriminator


def train(
		device=None,
		transform=None,
		generator=None,
		discriminator=None,
		batch_size=4,
		lr=0.0002,
		epochs=100,
):
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if transform is None:
		transform = T.Compose([
			T.ToTensor(),
			T.Normalize((0.5,), (0.5,))
		])
	if generator is None:
		generator = Generator().to(device)
	if discriminator is None:
		discriminator = Discriminator().to(device)
	min_loss_D = float('inf')
	min_loss_G = float('inf')
	train_dataset = MNIST(root='./data', train=True, download=True, transform=T.ToTensor())
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	criterion = nn.BCEWithLogitsLoss()
	optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
	optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
	for epoch in range(epochs):
		for real_images, _ in tqdm(
				train_loader,
				desc=f"Epoch {epoch+1}/{epochs}, min_loss_D={min_loss_D:.4f}, min_loss_G={min_loss_G:.4f}",
		):
			real_images = real_images.to(device)

			seed = torch.randn(batch_size, generator.latent_dim).to(device)
			fake_images = generator(seed)

			real_labels = torch.ones(batch_size, 1).to(device)
			fake_labels = torch.zeros(batch_size, 1).to(device)

			# Train Discriminator
			optimizer_D.zero_grad()

			outputs_real = discriminator(real_images)
			outputs_fake = discriminator(fake_images.detach())

			loss_real = criterion(outputs_real, real_labels)
			loss_fake = criterion(outputs_fake, fake_labels)

			loss_D = loss_real + loss_fake
			loss_D.backward()
			optimizer_D.step()

			# Train Generator
			optimizer_G.zero_grad()

			outputs = discriminator(fake_images)
			loss_G = criterion(outputs, real_labels)

			loss_G.backward()
			optimizer_G.step()

			# Save the best model
			if loss_D < min_loss_D:
				min_loss_D = loss_D
				torch.save(discriminator.state_dict(), './weights/discriminator.pth')
			if loss_G < min_loss_G:
				min_loss_G = loss_G
				torch.save(generator.state_dict(), './weights/generator.pth')



