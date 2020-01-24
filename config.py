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
	device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
	batch_size = 4
	is_shuffle_for_train_data = False
	train_epoch = 5

	input_width = 512
	input_height = 512
	output_width = 512
	output_height = 512

	cos_dice_loss_factor = 1.7

	learning_rate = 0.005

	case_strs = [
		'00', '01', '02', '03', '04', '05', '06', '07', '08', '09'
	] + [str(i) for i in range(10, 50)]

	@classmethod
	def to_int_tensor(cls, ndarray):
		ans = torch.from_numpy(ndarray)
		return ans.int()
