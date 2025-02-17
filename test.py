from unittest import TestCase

from networks.gan.train import train


class Test(TestCase):
	def test_gan(self):
		train()
