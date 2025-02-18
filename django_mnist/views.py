import base64
import io
import json
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from torch.nn import functional as F

from networks.cnn import CNN
from networks.gan import Generator, Discriminator


device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load('./weights/cnn.pth'))
cnn_model.eval()

generator_model = Generator().to(device)
generator_model.load_state_dict(torch.load('./weights/generator.pth'))
generator_model.eval()

discriminator_model = Discriminator().to(device)
discriminator_model.load_state_dict(torch.load('./weights/discriminator.pth'))
discriminator_model.eval()

seperated_generator_models = [Generator().to(device) for _ in range(10)]
[seperated_generator_models[i].load_state_dict(torch.load(f'./weights/generator_{i}.pth')) for i in range(10)]
[seperated_generator_models[i].eval() for i in range(10)]

seperated_discriminator_models = [Discriminator().to(device) for _ in range(10)]
[seperated_discriminator_models[i].load_state_dict(torch.load(f'./weights/discriminator_{i}.pth')) for i in range(10)]
[seperated_discriminator_models[i].eval() for i in range(10)]


def hello(request):
	return HttpResponse("Hello, World!")


@csrf_exempt
def cnn_process(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'Method not allowed'}, status=405)
	body_str = request.body.decode('utf-8')
	body = json.loads(body_str)
	image = base642tensor(body['image'])
	soft_label = cnn_model(image)
	soft_label = F.softmax(soft_label)
	predicted = torch.argmax(soft_label)
	pos_prob = discriminator_model(image)  # 图片非伪造的概率
	soft_label = soft_label.tolist()
	predicted = predicted.item()
	pos_prob = pos_prob.item()
	return JsonResponse({
		'pos_prob': pos_prob,
		'predicted': predicted,
		'soft_label': soft_label,
	})


@csrf_exempt
def gan_process(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'Method not allowed'}, status=405)
	seed = torch.randn(1, generator_model.latent_dim).to(device)
	fake_images = [seperated_generator_models[i](seed) for i in range(10)]
	pos_probs = [discriminator_model(fake_images[i]) for i in range(10)]
	fake_images = [tensor2base64(fake_images[i]) for i in range(10)]
	pos_probs = [pos_probs[i].item() for i in range(10)]
	return JsonResponse({
		'fake_images': fake_images,
		'pos_probs': pos_probs,
	})


def base642tensor(base64_str):
	if base64_str.startswith("data:image/png;base64,"):
		base64_str = base64_str[len("data:image/png;base64,"):]
	image_bytes = base64.b64decode(base64_str)
	image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 将图像转换为灰度图
	image = image.resize((28, 28))  # 调整图像大小为 28x28 像素
	image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0  # 将图像数据转换为 tensor 并归一化
	image_tensor = image_tensor.view(1, 1, 28, 28)  # 调整 tensor 形状以匹配模型输入
	image_tensor = 1.0 - image_tensor  # 反转图像像素值，满足模型输入要求
	return image_tensor.to(device)


def tensor2base64(tensor):
	image = tensor.cpu().detach().numpy()
	image = image.reshape(28, 28)
	image = (1.0 - image) * 255.0
	image = image.astype(np.uint8)
	image = Image.fromarray(image)
	buffer = io.BytesIO()
	image.save(buffer, format='PNG')
	image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
	return "data:image/png;base64," + image_base64


def index(request):
	context = {'hello': "Hello, Django!" + datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
	return render(request, 'index.html', context)


def cnn(request):
	return render(request, 'cnn.html')


def gan(request):
	return render(request, 'gan.html')
