from matplotlib import pyplot as plt
from torch import Tensor
import time


def save_one_slice(one_slice: Tensor, name: str):
	path = r'F:\71118123\winter_holiday\ai\imgs\{}.jpg'.format(name)
	plt.imsave(path, one_slice.cpu())


def show_one_slice(one_slice: Tensor, title: str):
	plt.title(title)
	plt.imshow(one_slice.cpu())
	plt.show()


def show_a_batch(batch_data: Tensor, title: str = ''):
	for i, one_data in enumerate(batch_data):
		one_data = one_data.squeeze(dim=0)
		show_one_slice(one_data, title + '.{}'.format(i))


def get_local_time_str(fmt_str: str = '%X'):
	local_time = time.localtime()
	local_time_str = time.strftime(
		fmt_str, local_time
	)
	return local_time_str
