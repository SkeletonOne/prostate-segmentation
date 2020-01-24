from model.unet_models import UNet
from model.dice_loss_models import CosDiceLoss
from processor import *
from helper import *
from torch import nn as nn
import torch
from tqdm import tqdm
from torch.autograd import Variable


def train_and_save():
	"""在这个函数中完成训练以及得到的模型的保存."""
	# Step1: 获取到数据
	print('get data set'.center(31, '*'))
	train_dataset = ProstateSegedSliceDataset()
	train_data_loader = DataLoader(
		dataset=train_dataset,
		batch_size=TorchConfig.batch_size,
		shuffle=TorchConfig.is_shuffle_for_train_data
	)
	print()

	# Step2: 配置模型
	print('config unet model'.center(31, '*'))
	unet_model = UNet(1, 512)
	unet_model.to(TorchConfig.device)

	# Step3: 配置损失函数
	unet_model.set_loss_function(CosDiceLoss(TorchConfig.cos_dice_loss_factor))

	# Step4: 配置优化器
	optimizer = torch.optim.RMSprop(unet_model.parameters(), lr=TorchConfig.learning_rate)
	unet_model.set_optimizer(optimizer)
	print()

	# Step5: 开始训练
	print('start train'.center(31, '*'))
	unet_model.train()
	for epoch_index in range(1, TorchConfig.train_epoch + 1):
		for train_batch_index in range(1, len(train_data_loader) + 1):
			x_train: Variable
			y_train: Variable
			x_train_tensor: Tensor
			y_train_tensor: Tensor
			x_train_tensor, y_train_tensor = train_data_loader[train_batch_index]
			x_train = Variable(x_train_tensor).to(TorchConfig.device)
			y_train = Variable(y_train_tensor).to(TorchConfig.device)
			# TODO 继续完成训练部分的代码
	print()
	print('end...')


if __name__ == '__main__':
	train_and_save()
