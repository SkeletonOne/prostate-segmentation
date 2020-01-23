from pathlib import Path
import os
import torch


class PathConfig(object):
	project_dir = Path(r'F:\Research\prostate-segmentation')
	data_dir = project_dir / 'data'
	test_data_dir = data_dir / 'test'
	train_data_dir = data_dir / 'train'

	raw_mhd_format_str = r'Case%s.mhd'
	seg_mhd_format_str = r'Case%s_segmentation.mhd'

	@classmethod
	def get_train_file_path(cls, case_str: str, is_seg: bool = False):
		file_name = (
			(cls.raw_mhd_format_str % case_str) if not is_seg else (
				cls.seg_mhd_format_str % case_str
			)
		)
		file_path = cls.train_data_dir / file_name
		if not os.path.exists(file_path):
			raise KeyError('文件路径 %s 不存在' % file_path.absolute())
		return file_path

	@classmethod
	def get_test_file_path(cls, case_str: str):
		file_name = cls.raw_mhd_format_str % case_str
		file_path = cls.test_data_dir / file_name
		if not os.path.exists(file_path):
			raise KeyError('文件路径 %s 不存在' % file_path.absolute())
		return file_path


class TorchConfig(object):
	device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
	batch_size = 8
	is_shuffle_for_train_data = False
	train_epoch = 5

	@classmethod
	def to_tensor(cls, ndarray):
		ans = torch.from_numpy(ndarray)
		return ans.to(device=cls.device)
