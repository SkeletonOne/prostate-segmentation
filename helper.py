from matplotlib import pyplot as plt
from torch import Tensor


def show_one_slice(one_slice: Tensor, title: str):
	plt.title(title)
	plt.imshow(one_slice.cpu())
	plt.show()
