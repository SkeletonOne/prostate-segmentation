from pathlib import Path
import os
from config import *
import SimpleITK as sitk
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from helper import *
from torch.nn import functional as f


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
		_case_str: str, is_seg: bool = False
	):
		mhd_file_path: Path
		if the_dir is PathConfig.train_data_dir:
			mhd_file_path = PathConfig.get_train_file_path(_case_str, is_seg)
		elif the_dir is PathConfig.test_data_dir:
			mhd_file_path = PathConfig.get_test_file_path(_case_str)
		else:
			raise KeyError('the_dir %s is illegal' % the_dir.absolute())
		itk_image = sitk.ReadImage(str(mhd_file_path.absolute()))
		numpy_image = sitk.GetArrayFromImage(itk_image)
		original_coordinate = np.array(list(itk_image.GetOrigin()))  # CT 原点坐标
		space = np.array(list(itk_image.GetSpacing()))  # CT 像素间隔
		return CTScan(
			TorchConfig.to_int_tensor(numpy_image),
			TorchConfig.to_int_tensor(original_coordinate),
			TorchConfig.to_int_tensor(space)
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

	@classmethod
	def pad_to_512_512(cls, tensor: Tensor):
		delta1 = 512 - tensor.shape[-2]
		left1, right1 = 0, 0
		if delta1 % 2 == 0:
			left1 = right1 = delta1 / 2
		else:
			left1 = delta1 // 2
			right1 = left + 1

		delta2 = 512 - tensor.shape[-1]
		left2, right2 = 0, 0
		if delta2 % 2 == 0:
			left2 = right2 = delta2 / 2
		else:
			left2 = delta2 // 2
			right2 = left2 + 1

		pad = [int(left1), int(right1), int(left2), int(right2)]

		ans = f.pad(tensor, pad)

		return ans

	@classmethod
	def pad_to_n_n_for_last_2_dim(cls, tensor: Tensor, n: int):
		if tensor.shape[-1] != tensor.shape[-2]:
			raise KeyError('当前需要被 pad 的 tensor 的大小不为正方形 {}'.format(tensor.shape))
		size = int(tensor.shape[-1])
		if size == n:
			return tensor
		left_delta: int
		right_delta: int
		ans_tensor: Tensor
		if size > n:
			delta = size - n
			if delta % 2 == 0:
				left_delta = right_delta = int(delta / 2)
			else:
				left_delta = int(delta // 2)
				right_delta = left_delta + 1
			ans_tensor = f.pad(
				tensor, [-left_delta, -right_delta, -left_delta, -right_delta]
			)
		else:
			delta = n - size
			if delta % 2 == 0:
				left_delta = right_delta = int(delta / 2)
			else:
				left_delta = int(delta // 2)
				right_delta = left_delta + 1
			ans_tensor = f.pad(
				tensor, [left_delta, right_delta, left_delta, right_delta]
			)
		return ans_tensor


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

		# Step6: 转换所有
		self.train_slices = self.trans_all_to_3_dim()

	def __getitem__(self, _index: int):
		return self.train_slices[_index][0], self.train_slices[_index][1]

	def __len__(self) -> int:
		return len(self.train_slices)

	def trans_all_to_3_dim(self):
		"""将所有得图片都变成 C * W * H. (1 * 512 * 512)"""
		temp = []
		for train_slice_tuple in self.train_slices:
			t = []
			for index, s in enumerate(train_slice_tuple):
				s: Tensor
				s = CTScan.pad_to_n_n_for_last_2_dim(s, 512)
				temps = s.unsqueeze(0)
				t.append(temps)
			t = tuple(t)
			temp.append(t)
		return temp


if __name__ == '__main__':
	# count = 0
	# for case_str in TorchConfig.case_strs:
	# 	seg_ct_scan_slice = CTScan.get_one_slice(
	# 		PathConfig.train_data_dir,
	# 		case_str, is_seg=True,
	# 		slice_index=0
	# 	)
	# 	if seg_ct_scan_slice.shape[-1] <= 324:
	# 		print('case: {}, shape: {}'.format(case_str, seg_ct_scan_slice.shape))
	# 		count += 1
	# print('total = {}'.format(count))
	# case_string = '34'
	# ct_scan = CTScan.load_ct_torch_from_mhd(
	# 	PathConfig.train_data_dir, case_string,
	# 	True
	# )
	# seg_slices = ct_scan.get_slices()
	# seg_slices = [CTScan.pad_to_n_n_for_last_2_dim(s, 324) for s in seg_slices]
	#
	# ct_scan = CTScan.load_ct_torch_from_mhd(
	# 	PathConfig.train_data_dir, case_string,
	# 	False
	# )
	# raw_slices = ct_scan.get_slices()
	# raw_slices = [CTScan.pad_to_n_n_for_last_2_dim(s, 512) for s in raw_slices]
	#
	# for index in range(6, 17, 1):
	# 	# print(
	# 	# 	'{}.raw.{}.shape = {} && {}.seg.{}.shape = {}'.format(
	# 	# 		case_string, index, raw_slices[index].shape, case_string, index, seg_slices[index].shape
	# 	# 	)
	# 	# )
	# 	show_one_slice(raw_slices[index], '{}.raw.{}'.format(case_string, index))
	# 	show_one_slice(seg_slices[index], '{}.seg.{}'.format(case_string, index))
	pass
