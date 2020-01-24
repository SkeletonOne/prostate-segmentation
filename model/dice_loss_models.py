import torch
import torch.nn as nn
from processor import *
import config


def cal_DSC(predicts, targets):
	batch_size = int(targets.shape[0])
	smooth = 1
	input_flat = predicts.view((batch_size, -1))
	target_flat = targets.view((batch_size, -1))
	intersection = input_flat * target_flat  # 求交集
	i_sum = input_flat.sum(dim=1)
	t_sum = target_flat.sum(dim=1)
	inter_sum = intersection.sum(dim=1)
	_a = 2.0 * inter_sum + smooth
	_b = i_sum + t_sum + smooth
	ans = _a.float() / _b.float()
	return ans


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
		dsc = cal_DSC(predicts, targets)
		return 1 - dsc


class CosDiceLoss(nn.Module):

	def __init__(self, q_factor: float):
		super(CosDiceLoss, self).__init__()
		if q_factor <= 1.0:
			raise KeyError('Q factor must > 1.0')
		self.q_factor = q_factor

	def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
		dsc = cal_DSC(predicts, targets)
		import numpy as np
		temp = (dsc * (np.pi / 2)).float()
		t = torch.cos(temp)
		ans = torch.pow(t, self.q_factor)
		return ans


if __name__ == '__main__':
	# train_dataset = ProstateSegedSliceDataset()
	# train_data_loader = DataLoader(
	# 	dataset=train_dataset,
	# 	batch_size=TorchConfig.batch_size,
	# 	shuffle=TorchConfig.is_shuffle_for_train_data
	# )
	# a = None
	# b = None
	# for batch_index, batch_data in enumerate(train_data_loader):
	# 	if batch_index < 5:
	# 		continue
	# 	if batch_index == 5:
	# 		a = batch_data[1]
	# 		continue
	# 	if batch_index == 6:
	# 		b = batch_data[1]
	# 		break
	# # print(a.shape)
	# d = CosDiceLoss(1.7)
	# # show_a_batch(a, 'a')
	# # show_a_batch(b, 'b')
	# loss = d(a, b)
	# print(loss)
	pass
