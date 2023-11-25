import json
import os
import gc
import sys

from pyparsing import OnlyOnce
sys.path.append('/home/rgu/Documents/GitHub/S3_knee_segmentation/ct_all_preprocess')
import nibabel as nib
import numpy as np
import ants
import re
import glob
import random
import pickle
from sklearn.decomposition import PCA
from utils import check_path_exist

def compute_min_max(label, execpt_axis):
    indices = np.nonzero(label.sum(axis=execpt_axis) > 0)[0]
    axis_min = max(0, indices.min() - 10)
    axis_max = min(label.shape[0], indices.max() + 10)
    return axis_min, axis_max

def extract_one_bone(full_leg_seg, label):
	# x_min, x_max = compute_min_max(mask, (1, 2))
	# y_min, y_max = compute_min_max(mask, (0, 2))
	# z_min, z_max = compute_min_max(mask, (0, 1))
	# cropped_new_arr = mask[x_min:x_max, y_min:y_max, z_min:z_max]
	mask_img = ants.utils.get_mask(full_leg_seg,1,2)
	fixed_cropped = ants.utils.crop_image(mask_img, mask_img, 1)
	cropped_padded = ants.utils.pad_image(fixed_cropped,pad_width=[(20, 20), (20, 20), (20, 20)])
	return cropped_padded

@OnlyOnce
def save_training_json(moving_train_dir, json_path):
    left_leg_list_uk = []
    left_leg_list_us = []

	pattern_4_digits = r'halvleg_(\d{4})\.nii\.gz'
	pattern_6_digits_starting_with_2 = r'halvleg_(2\d{5})\.nii\.gz'
 	for filename in os.listdir(moving_train_dir):
		if re.match(pattern_4_digits, filename):
			left_leg_list_uk.append(os.path.join(moving_train_dir, filename))
		elif re.match(pattern_6_digits_starting_with_2, filename):
			left_leg_list_us.append(os.path.join(moving_train_dir, filename))
	
	left_leg_list = left_leg_list_uk + left_leg_list_us
	all_files = set(os.listdir(moving_train_dir))
	left_set = set([os.path.basename(f) for f in left_leg_list])
	right_leg_set = all_files - left_set
	right_leg_list = [os.path.join(moving_train_dir, f) for f in right_leg_set]
 
 	with open(json_path, 'w') as f:
		json.dump({'left_leg_list': left_leg_list, 'right_leg_list': right_leg_list}, f)
	


class RecSSM(object):
	def __init__(self, numOfImg4SSM=30, shape=None):
		self.numOfImg4SSM = numOfImg4SSM
		if shape is None:
			self.fix_img_x = 256
			self.fix_img_y = 512
			self.fix_img_z = 1500
		else:
			if len(shape) == 3:
				pass
			else:
				raise ValueError("only accept 3D volumetric input")
			self.fix_img_x, self.fix_img_y, self.fix_img_z = shape


	def reg(self, fix_img, moving_img):
		outs = ants.registration(fix_img, moving_img, type_of_transforme='rigid')
		warped_img = outs['warpedmovout']
		return outs

	
	def inverse_reg(self, fixed1, moving1, moving2):
		outs1 = ants.registration(fixed1, moving1, type_of_transforme='rigid')
		outs2 = ants.apply_transforms(moving1, moving2, transformlist=outs1['invtransforms'])
		outs2 = outs2.numpy()
		outs2 = (outs2 > 0)
		outs2 = outs2 + 1 - 1
		return outs2

	def ssm_train(self, warped_train_dir):
		complete = glob.glob(os.path.join(warped_train_dir, '*.nii.gz'))
		# shuffle the list of complete
		random.shuffle(complete)
		pca = PCA(n_components=len(complete))
		# (x,y,z) is the median shape calculated by nnUNet
		data = np.zeros(shape=(len(complete), self.fix_img_x, self.fix_img_y, self.fix_img_z), dtype="int16")
		for i in range(len(complete)):
			temp = nib.load(complete[i]).get_fdata()
			data[i, :, :, :] = temp
			del temp
		data_ = data[0:self.numOfImg4SSM]
		self.mean_shape = data_.mean(axis=0)
		data = np.reshape(data, (len(complete), self.fix_img_x * self.fix_img_y * self.fix_img_z))
		data_pca = pca.fit_transform(data)

		data_inv = np.linalg.pinv(data)

		del data
		gc.collect()

		self.eigenvec = data_inv.dot(data_pca)

		# del data_
		del data_inv
		gc.collect()

	def ssm_test(self, testImg, useOnlyMeanShape=False):
		testdata = np.reshape(testImg, (1, self.fix_img_x * self.fix_img_y * self.fix_img_z))
		testdatapca = testdata.dot(self.eigenvec)

		lambda_n = []
		for i in range(len(testdatapca)):
			lambda_n.append(testdatapca[i])
		lambda_n = np.array(lambda_n)
		# scale [0,1]
		lambda_n = (lambda_n - np.min(lambda_n)) / np.ptp(lambda_n)
		lambda_n = np.transpose(lambda_n)
		reconstructed = self.eigenvec.dot(lambda_n)
		reconstructed = np.reshape(reconstructed, (self.fix_img_x, self.fix_img_y, self.fix_img_z))
		if useOnlyMeanShape:
			rec = self.mean_shape
		else:
			rec = reconstructed + self.mean_shape

		rec = (rec > 0)
		rec = rec + 1 - 1

		return rec


if __name__ == "__main__":
    bone_structure = ['left_femur', 'right_femur', 'left_tibia', 'right_tibia']
    for i in range(len(bone_structure)):
        moving_train_dir = '/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr'
        moving_test_dir = '/home/rgu/Documents/ssm_testing_samples'
        warped_train_dir = '/home/rgu/Documents/wrapped_label/' + bone_structure[i] + '/train/'
        warped_test_dir = '/home/rgu/Documents/wrapped_label/' + bone_structure[i] + '/test/'
        results_dir = '/home/rgu/Documents/ssm_results/' + bone_structure[i] + '/'
        json_path = os.path.join('/home/rgu/Documents/ssm_results', 'training.json')
        
        check_path_exist(warped_train_dir)
        check_path_exist(warped_test_dir)
        check_path_exist(results_dir)
        
        save_training_json(moving_train_dir, json_path)

		with open(json_path, 'r') as f:
			training_dict = json.load(f)
		if 'left' in bone_structure[i]:
			leg_list = training_dict['left_leg_list']
			print('generating fixed image...')
			fixed_img = ants.image_read('/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr/halvleg_1064.nii.gz')
			fixed_anatomy = extract_one_bone(fixed_img, 1)
			ants.image_write(fixed_anatomy,'/home/rgu/Documents/' + bone_structure[i] + '.nii.gz')
			del fixed_anatomy
			gc.collect()
		elif 'right' in bone_structure[i]:
			leg_list = training_dict['right_leg_list']
			print('generating fixed image...')
			fixed_img = ants.image_read('/home/rgu/Documents/UK dataset/nnUNet_raw/Dataset1000_NMDID/labelsTr/halvleg_62.nii.gz')
			fixed_anatomy = extract_one_bone(fixed_img, 1)
			ants.image_write(fixed_anatomy,'/home/rgu/Documents/' + bone_structure[i] + '.nii.gz')
			del fixed_anatomy
			gc.collect()

		#  because two dataset is mixed, the fixed image is calculated by averaging N images from two datasets
		# fixed_list = random.sample(left_leg_list_uk, 3) + random.sample(left_leg_list_us, 3)
		# fixed_img = ants.average_images(fixed_list)


		# to reload the one reference shape
		fixed_anatomy = ants.image_read('/home/rgu/Documents/left_femur.nii.gz')
		model = RecSSM(30, fixed_anatomy.shape)


		print('warping training images...')
		for filename_tr in leg_list:
			moving_img = ants.image_read(filename_tr)
			moving_mask_img = ants.utils.get_mask(moving_img, 1, 2)
			moving_left_femur = ants.utils.crop_image(moving_mask_img, moving_mask_img)
			outs = model.reg(fixed_anatomy, moving_left_femur)
			warped_img = outs['warpedmovout']
			ants.image_write(warped_img, warped_train_dir + os.path.basename(filename_tr))
			print(os.path.basename(filename_tr) + ' is done')


		print('building SSM...')
		model.ssm_train(warped_train_dir)
		
		if os.path.exists(os.path.join(results_dir, 'ssm_model.pkl')):
			print('loading SSM...')
   			with open(os.path.join(results_dir, 'ssm_model.pkl'), 'rb') as f:
				model = pickle.load(f)
		else:
			with open(os.path.join(results_dir, 'ssm_model.pkl'), 'wb') as f:
				pickle.dump(model, f)


		print('warping test images...')
		for filename_ts in os.listdir(moving_test_dir):
			moving_img = ants.image_read(os.path.join(moving_test_dir, filename_ts))
			outs = model.reg(fixed_anatomy, moving_img)
			warped_img = outs['warpedmovout']
			ants.image_write(warped_img, warped_test_dir + filename_ts)
			print(filename_ts + ' is done')
		
  
		print('fitting...')
		for filename_wrap in os.listdir(warped_test_dir):
			test = nib.load(os.path.join(warped_test_dir, filename_wrap))
			test_data = test.get_fdata().astype(np.int16)
			rec = model.ssm_test(test_data, useOnlyMeanShape=False)
			implant = rec-test_data
		
			rec_img = nib.Nifti1Image(rec, test.affine, test.header)
			nib.save(rec_img, os.path.join(results_dir, 'rec_'+filename_wrap))
			implant_img = nib.Nifti1Image(implant, test.affine, test.header)
			nib.save(implant_img, os.path.join(results_dir, 'implant_'+filename_wrap))


		print('converting the results back to original image space...')
		for filename_wrap in os.listdir(moving_test_dir):
			moving = ants.image_read(os.path.join(moving_test_dir, filename_wrap))
			moving_rec = ants.image_read(os.path.join(results_dir, 'rec_'+filename_wrap))
			converted_img = model.inverse_reg(fixed_anatomy, moving, moving_rec)
			moving_nib = nib.load(os.path.join(moving_test_dir, filename_wrap))
			restore_img = nib.Nifti1Image(converted_img.astype(np.int16), moving_nib.affine, moving_nib.header)
			nib.save(restore_img, os.path.join(results_dir, filename_wrap))
