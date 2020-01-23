from pathlib import Path
import os
from config import *
import SimpleITK as sitk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from helper import *


class CTScan(object):

	def __init__(self, ct_img: torch.Tensor, cor: torch.Tensor, space: torch.Tensor):
		self.__ct_img = ct_img
		self.__coordinate = cor
		self.__space = space

	@property
	def get_ct_scan_imgs(self):
		return self.__ct_img

	@property
	def get_coordinate(self):
		return self.__coordinate

	@property
	def get_space(self):
		return self.__space

	def get_slices(self):
		_ans = list(self.get_ct_scan_imgs)
		return _ans

	@classmethod
	def load_ct_torch_from_mhd(
		cls, the_dir: Path,
		case_str: str, is_seg: bool = False
	):
		mhd_file_path: Path
		if the_dir is PathConfig.train_data_dir:
			mhd_file_path = PathConfig.get_train_file_path(case_str, is_seg)
		elif the_dir is PathConfig.test_data_dir:
			mhd_file_path = PathConfig.get_test_file_path(case_str)
		else:
			raise KeyError('the_dir %s is illegal' % the_dir.absolute())
		itk_image = sitk.ReadImage(str(mhd_file_path.absolute()))
		numpy_image = sitk.GetArrayFromImage(itk_image)
		original_coordinate = np.array(list(itk_image.GetOrigin()))  # CT 原点坐标
		space = np.array(list(itk_image.GetSpacing()))  # CT 像素间隔
		return CTScan(
			TorchConfig.to_tensor(numpy_image),
			TorchConfig.to_tensor(original_coordinate),
			TorchConfig.to_tensor(space)
		)

	@classmethod
	def get_one_slice(
		cls,
		the_dir: Path, case_str: str, is_seg: bool = False,
		slice_index: int = 0
	):
		ct_scan = CTScan.load_ct_torch_from_mhd(the_dir, case_str, is_seg)
		slices = ct_scan.get_slices()
		return slices[slice_index]


class ProstateSegedSliceDataset(Dataset):
	train_data_dir = PathConfig.train_data_dir

	def __init__(self):
		super().__init__()
		# Step1: 排序后的未分割
		raw_mhd_file_names = [
			str(mhd_file_name)
			for mhd_file_name in os.listdir(self.train_data_dir) if (
				('.mhd' in str(mhd_file_name)) and
				('_segmentation' not in str(mhd_file_name))
			)
		]
		raw_mhd_file_names.sort()

		# Step2: 排序后的有标注的
		seged_mhd_file_names = [
			str(mhd_file_name)
			for mhd_file_name in os.listdir(self.train_data_dir) if (
				('_segmentation.mhd' in str(mhd_file_name))
			)
		]
		seged_mhd_file_names.sort()

		# Step3: 判断是否相等, 并且获取到所有的 case_str
		if len(seged_mhd_file_names) != len(raw_mhd_file_names):
			raise KeyError('训练集的个数不正常')
		n = len(seged_mhd_file_names)
		pattern = re.compile(r'(\d)+')
		case_strs = []
		for i in range(n):
			raw_mhd_file_name = raw_mhd_file_names[i]
			seged_mhd_file_name = seged_mhd_file_names[i]
			raw_case_str = pattern.search(raw_mhd_file_name).group()
			seged_case_str = pattern.search(seged_mhd_file_name).group()
			if (
				(raw_case_str is None) or (seged_case_str is None) or
				(raw_case_str != seged_case_str)
			):
				raise KeyError('训练集中的内容不匹配')
			case_strs.append(raw_case_str)

		# Step4: 收集 CTScan 对象
		self.train_ct_scan_tuples = []
		for case_str in case_strs:
			raw_ct_scan = CTScan.load_ct_torch_from_mhd(
				self.train_data_dir,
				case_str, False
			)
			seged_ct_scan = CTScan.load_ct_torch_from_mhd(
				self.train_data_dir,
				case_str, True
			)
			self.train_ct_scan_tuples.append((raw_ct_scan, seged_ct_scan))

		# Step5: 收集 slices
		rs = []
		ss = []
		for r, s in self.train_ct_scan_tuples:
			raw_slices = r.get_slices()
			seged_slices = s.get_slices()
			if len(raw_slices) != len(seged_slices):
				raise KeyError('切片数目不对应')
			rs.extend(raw_slices)
			ss.extend(seged_slices)
		self.train_slices = list(zip(rs, ss))

	def __getitem__(self, index: int):
		return self.train_slices[index]

	def __len__(self) -> int:
		return len(self.train_slices)


if __name__ == '__main__':
	pass
