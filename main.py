from model.unet_models import UNet
from model.dice_loss_models import CosDiceLoss, cal_dice_accuracy, cal_average_loss
from processor import *
from helper import *
from torch import nn as nn
import torch
from tqdm import tqdm
from torch.autograd import Variable
import sys


def train_and_save():
	"""在这个函数中完成训练以及得到的模型的保存."""
	# Step0: 获取当前的时间字符串
	local_time_string = get_local_time_str('%Y%m%d%H%M%S')
	PathConfig.create_log_file(local_time_string)

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
	loss_function_model = CosDiceLoss(TorchConfig.cos_dice_loss_factor)
	loss_function_model.to(TorchConfig.device)
	unet_model.set_loss_function(loss_function_model)

	# Step4: 配置优化器
	optimizer = torch.optim.RMSprop(unet_model.parameters(), lr=TorchConfig.learning_rate)
	unet_model.set_optimizer(optimizer)
	print()

	# Step5: 开始训练
	print('start train'.center(31, '*'))
	best_loss = sys.maxsize
	PathConfig.write_to_log_file(
		local_time_string,
		'\r\nStart training when {}.\r\n'.format(
			get_local_time_str()
		)
	)
	unet_model.train()
	for epoch_index in range(1, TorchConfig.train_epoch + 1):
		for train_batch_index, train_batch_data in enumerate(train_data_loader):
			# 训练
			_train_batch_index = train_batch_index + 1
			x_train: Variable
			y_train: Variable
			x_train_tensor: Tensor
			y_train_tensor: Tensor
			x_train_tensor, y_train_tensor = train_batch_data
			x_train_tensor = x_train_tensor.to(TorchConfig.device)
			y_train_tensor = y_train_tensor.to(TorchConfig.device)
			x_train = Variable(x_train_tensor).to(TorchConfig.device)
			y_train = Variable(y_train_tensor).to(TorchConfig.device)
			y_pred, loss = unet_model.train_one_batch(x_train, y_train)
			mean_loss = loss.item()
			accuracy = cal_dice_accuracy(y_pred, y_train_tensor)
			# 打印结果
			prompt_str = (
				'epoch: {0}, batch: {1}, mean loss: {2:.4f}, accuracy: {3:.4f}'.format(
					epoch_index, _train_batch_index,
					mean_loss, accuracy
				)
			)
			print(prompt_str)
			PathConfig.write_to_log_file(
				local_time_string, prompt_str
			)
			# 判断是否要保存
			if mean_loss < best_loss:
				best_loss = mean_loss
				# 保存模型
				PathConfig.save_model(unet_model, local_time_string, '{:.3f}'.format(mean_loss))
	print()
	print('end training...')


if __name__ == '__main__':
	train_and_save()
