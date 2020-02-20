from pathlib import Path
import os
import torch
import time
from helper import *


class PathConfig(object):
	project_dir = Path(r'F:\Research\prostate-segmentation')
	data_dir = project_dir / 'data'
	test_data_dir = data_dir / 'test'
	train_data_dir = data_dir / 'train'
	output_dir = project_dir / 'output'
	output_model_dir = output_dir / 'model'
	log_file_dir = project_dir / 'log'

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

	@classmethod
	def get_log_file_path_str(cls, local_time_str: str) -> str:
		log_file_path = PathConfig.log_file_dir / (local_time_str + '.txt')
		log_file_path_str = str(log_file_path.absolute())
		return log_file_path_str

	@classmethod
	def create_log_file(cls, local_time_str: str):
		log_file_path_str = PathConfig.get_log_file_path_str(local_time_str)
		with open(
			log_file_path_str, mode='w+', encoding='utf-8'
		) as out_file:
			out_file.write(
				'Create this log txt when {}.\r\n'.format(
					get_local_time_str()
				)
			)

	@classmethod
	def write_to_log_file(cls, local_time_str: str, line_info: str):
		log_path = PathConfig.get_log_file_path_str(local_time_str)
		with open(log_path, mode='a+', encoding='utf-8') as out_file:
			line_info = line_info + ('' if line_info.endswith('\r\n') else '\r\n')
			out_file.write(line_info)

	@classmethod
	def save_model(cls, model: torch.nn.Module, local_time_str: str, loss_str: str):
		"""完成对于模型的保存."""
		import os
		temp_state = model.state_dict()

		# 创建目录
		save_path = local_time_str + '_' + loss_str + '.pkl'
		save_path = PathConfig.output_model_dir / save_path
		save_path_str = str(save_path.absolute())
		# if not os.path.exists(save_path_str):
		# 	os.mkdir(save_path_str)

		# 将保存文件写入log
		PathConfig.write_to_log_file(
			local_time_str,
			('save model (loss is {}) when {}\r\n' +
				'loss is {}\r\n' +
				'input size is {} and output size is {}\r\n' +
				'epoch is {}\r\n' +
				'train batch size is {}\r\n\r\n').format(
				loss_str, get_local_time_str(),
				loss_str,
				TorchConfig.input_height, TorchConfig.output_height,
				TorchConfig.train_epoch,
				TorchConfig.batch_size
			)
		)

		# 保存
		torch.save(temp_state, save_path_str)


class TorchConfig(object):
	device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
	batch_size = 2
	is_shuffle_for_train_data = False
	train_epoch = 3

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

	@classmethod
	def to_float_tensor(cls, ndarray):
		return torch.from_numpy(ndarray).float()
