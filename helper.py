from matplotlib import pyplot as plt
from torch import Tensor


def show_one_slice(one_slice: Tensor, title: str):
	plt.title(title)
	plt.imshow(one_slice.cpu())
	plt.show()


def show_a_batch(batch_data: Tensor, title: str = ''):
	for i, one_data in enumerate(batch_data):
		one_data = one_data.squeeze(dim=0)
		show_one_slice(one_data, title + '.{}'.format(i))
