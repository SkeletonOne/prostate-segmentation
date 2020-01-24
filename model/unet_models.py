import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as f


class UNet(nn.Module):

	def __init__(self, img_channel: int, img_size: int, *args, **kwargs):
		super(UNet, self).__init__()
		# 输入的图片应该为: (batch_size, in_channels, img_size, img_size)
		# 在当前这个例子中, 我们是 (8, 1, 512, 512)
		self.in_channels = img_channel
		self.img_size = img_size
		# (bs, 1, 512, 512)
		self.double_conv_encode_1 = UNet.get_double_conv_kernel_3(
			self.in_channels, 64, kernel_size=3
		)
		# (bs, 64, 508, 508)
		self.max_pooling_2 = nn.MaxPool2d(
			kernel_size=2  # 步长与 kernel_size 一致
		)
		# (bs, 64, 254, 254)
		self.double_conv_encode_3 = UNet.get_double_conv_kernel_3(
			64, 128, kernel_size=3
		)
		# (bs, 128, 250, 250)
		self.max_pooling_4 = nn.MaxPool2d(
			kernel_size=2  # 步长与 kernel_size 一致
		)
		# (bs, 128, 125, 125)
		self.double_conv_encode_5 = UNet.get_double_conv_kernel_3(
			128, 256, kernel_size=3
		)
		# (bs, 256, 121, 121)
		self.max_pooling_6 = nn.MaxPool2d(
			kernel_size=2  # 步长与 kernel_size 一致
		)
		# (bs, 256, 60, 60)
		self.double_conv_encode_7 = UNet.get_double_conv_kernel_3(
			256, 512, kernel_size=3
		)
		# (bs, 512, 56, 56)
		self.max_pooling_8 = nn.MaxPool2d(
			kernel_size=2
		)
		# (bs, 512, 28, 28)
		self.double_conv_encode_and_decode_9 = (
			UNet.get_double_conv_kernel_3_with_conv_up_2(
				double_in_channels=512,
				double_out_channels=1024,
				up_out_channels=512
			)
		)
		# (bs, 512, 48, 48)
		# 这之后做一个拼接 (bs, 512, 56, 56)
		# (bs, 1024, 48, 48)
		self.double_conv_encode_and_decode_10 = (
			UNet.get_double_conv_kernel_3_with_conv_up_2(
				double_in_channels=1024,
				double_out_channels=512,
				up_out_channels=256
			)
		)
		# (bs, 256, 88, 88)
		# 这之后做一个拼接 (bs, 256, 121, 121)
		# (bs, 512, 88, 88)
		self.double_conv_encode_and_decode_11 = (
			UNet.get_double_conv_kernel_3_with_conv_up_2(
				double_in_channels=512,
				double_out_channels=256,
				up_out_channels=128
			)
		)
		# (bs, 128, 168, 168)
		# 这之后做一个拼接 (bs, 128, 250, 250)
		# (bs, 256, 168, 168)
		self.double_conv_encode_and_decode_12 = (
			UNet.get_double_conv_kernel_3_with_conv_up_2(
				double_in_channels=256,
				double_out_channels=128,
				up_out_channels=64
			)
		)
		# (bs, 64, 328, 328)
		# 这之后做一个拼接 (bs, 64, 508, 508)
		# (bs, 128, 328, 328)
		self.double_conv_decode_13 = UNet.get_double_conv_kernel_3(
			in_channels=128, out_channels=64
		)
		# (bs, 64, 324, 324)
		self.final_conv_14 = nn.Sequential(
			nn.Conv2d(
				in_channels=64, out_channels=1,
				kernel_size=1, stride=1, padding=0
			),
			nn.BatchNorm2d(1),
			nn.ReLU()
		)
		# (bs, 64, 324, 324)

		if 'loss_function' in kwargs:
			self.loss_function = kwargs['loss_function']
		else:
			self.loss_function = None

		if 'optimizer' in kwargs:
			self.optimizer = kwargs['optimizer']
		else:
			self.optimizer = None

	@classmethod
	def get_double_conv_kernel_3(
		cls,
		in_channels: int, out_channels: int,
		kernel_size: int = 3
	) -> nn.Module:
		ans = nn.Sequential(
			nn.Conv2d(
				in_channels, out_channels,
				kernel_size, stride=1,
				padding=0
			),
			nn.BatchNorm2d(
				out_channels
			),
			nn.ReLU(),
			nn.Conv2d(
				out_channels, out_channels,
				kernel_size, stride=1,
				padding=0
			),
			nn.BatchNorm2d(
				out_channels
			),
			nn.ReLU()
		)
		return ans

	@classmethod
	def get_double_conv_kernel_3_with_conv_up_2(
		cls,
		double_in_channels: int, double_out_channels: int,
		up_out_channels: int
	) -> nn.Module:
		ans = nn.Sequential(
			UNet.get_double_conv_kernel_3(
				double_in_channels, double_out_channels,
				kernel_size=3
			),
			nn.ConvTranspose2d(
				double_out_channels, up_out_channels,
				kernel_size=2, stride=2
			)
		)
		return ans

	@classmethod
	def crop_and_concat(
		cls,
		smaller_decoded: Tensor,
		larger_encoded: Tensor,
		need_crop: bool = True
	) -> Tensor:
		if need_crop:
			delta = larger_encoded.shape[2] - smaller_decoded.shape[2]
			left_delta: int
			right_delta: int
			if delta % 2 == 0:
				left_delta = right_delta = int(delta / 2)
			else:
				left_delta = int(delta // 2)
				right_delta = int(left_delta + 1)
			larger_encoded = f.pad(
				larger_encoded,
				[-left_delta, -right_delta, -left_delta, -right_delta]
			)
		# 对于 (bs, channel, height, width) 中的 channel 来拼接
		return torch.cat((smaller_decoded, larger_encoded), dim=1)

	def pad_to_img_size(self, tensor: Tensor):
		if tensor.shape[-1] != tensor.shape[-2]:
			raise KeyError('图片的维度宽高不一致 {} != {}'.format(tensor.shape[-2], tensor.shape[-1]))
		if tensor.shape[-1] > self.img_size:
			raise KeyError('最后一个维度是: {} > {}'.format(tensor.shape[-1], self.img_size))
		elif tensor.shape[-1] == self.img_size:
			return tensor
		else:
			delta = self.img_size - tensor.shape[-1]
			left_delta: int
			right_delta: int
			if delta % 2 == 0:
				left_delta = right_delta = int(delta / 2)
			else:
				left_delta = int(delta // 2)
				right_delta = int(left_delta + 1)
			ans = f.pad(tensor, [left_delta, right_delta, left_delta, right_delta])
			return ans

	def forward(self, inputs: Tensor):
		encode1 = self.double_conv_encode_1(inputs)
		encode2 = self.max_pooling_2(encode1)

		encode3 = self.double_conv_encode_3(encode2)
		encode4 = self.max_pooling_4(encode3)

		encode5 = self.double_conv_encode_5(encode4)
		encode6 = self.max_pooling_6(encode5)

		encode7 = self.double_conv_encode_7(encode6)
		encode8 = self.max_pooling_8(encode7)

		decode9 = self.double_conv_encode_and_decode_9(encode8)
		decode9 = UNet.crop_and_concat(
			decode9, encode7
		)

		decode10 = self.double_conv_encode_and_decode_10(decode9)
		decode10 = UNet.crop_and_concat(
			decode10, encode5
		)

		decode11 = self.double_conv_encode_and_decode_11(decode10)
		decode11 = UNet.crop_and_concat(
			decode11, encode3
		)

		decode12 = self.double_conv_encode_and_decode_12(decode11)
		decode12 = UNet.crop_and_concat(
			decode12, encode1
		)

		decode13 = self.double_conv_decode_13(decode12)

		decode14 = self.final_conv_14(decode13)
		ans = self.pad_to_img_size(decode14)
		return ans

	def set_loss_function(self, loss_function: nn.Module):
		self.loss_function = loss_function

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def eval_forward(self, x_inputs, y_targets):
		pred = self(x_inputs)
		loss = self.loss_function(pred, y_targets)
		return pred, loss

	def train_one_batch(self, x_train, y_train):
		"""完成一个 batch 的训练."""
		# 1. 首先前向传播, 计算一次
		y_pred, loss = self.eval_forward(x_train, y_train)
		# 2. 清空梯度
		self.optimizer.zero_grad()
		# 3. 反向传播计算梯度
		loss.backward()
		# 4. 优化器选择移动一个梯度的方位
		self.optimizer.step()
		# 5. 返回结果
		return y_pred, loss


if __name__ == '__main__':
	# inputs = torch.randn((2, 1, 512, 512))
	# print(inputs.shape)
	# unet = UNet(1, 512)
	# outputs = unet(inputs)
	# print(outputs.shape)
	pass
