from model import UNet
from processor import *


def train_and_save():
	"""在这个函数中完成训练以及得到的模型的保存."""
	# Step1: 获取到数据
	train_dataset = ProstateSegedSliceDataset()
	train_data_loader = DataLoader(
		dataset=train_dataset,
		batch_size=TorchConfig.batch_size,
		shuffle=TorchConfig.is_shuffle_for_train_data
	)

	# Step2: 配置模型
	



if __name__ == '__main__':
