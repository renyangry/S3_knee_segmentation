# ssm_model.py

import os
import glob
import random
import gc
import numpy as np
import nibabel as nib
import ants
from sklearn.decomposition import PCA


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
		outs = ants.registration(fix_img, moving_img, type_of_transforme='Affine') #QuickRigid /
		warped_img = outs['warpedmovout']
		return outs

	
	def inverse_reg(self, fixed1, moving1, moving2):
		outs1 = ants.registration(fixed1, moving1, type_of_transforme='Affine')
		outs2 = ants.apply_transforms(moving1, moving2, transformlist=outs1['invtransforms'])
		outs2 = outs2.numpy()
		# outs2[outs2 < 0] = 0
		outs2 = np.abs(outs2)
		# outs2_nib = nib.Nifti1Image(outs2, affine=np.eye(4))
		# nib.save(outs2_nib, os.path.join('/home/rgu/Documents/ssm_results/left_femur', 'reg1.nii.gz'))
  
		return outs2

	def ssm_train(self, warped_train_dir):
		complete = glob.glob(os.path.join(warped_train_dir, '*.nii.gz'))#[0:30]
		random.shuffle(complete)
		pca = PCA(n_components=len(complete))
		# (x,y,z) can be the median shape calculated by nnUNet
		data = np.zeros(shape=(len(complete), self.fix_img_x, self.fix_img_y, self.fix_img_z), dtype="int16")
		for i in range(len(complete)):
			temp = nib.load(complete[i]).get_fdata()
			data[i, :, :, :] = temp
			del temp
		data_ = data[0:self.numOfImg4SSM, :, :, :]
		self.mean_shape = data_.mean(axis=0)

		# debug
		# test = nib.Nifti1Image(self.mean_shape, affine=np.eye(4))
		# nib.save(test, os.path.join('/home/rgu/Documents/ssm_results/left_femur', 'ssm_meanshape.nii.gz'))
		# print('mean shape outputted for evaluation')
		#
		data = np.reshape(data, (len(complete), self.fix_img_x * self.fix_img_y * self.fix_img_z))
		data_pca = pca.fit_transform(data)

		# pseudo inverse
		data_inv = np.linalg.pinv(data)
		# eigenvector of original data = directions of maximum variance of original data
		self.eigenvec = data_inv.dot(data_pca)

		del data
		del data_
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

		# rec[rec < 0] = 0 # negative values are not the usual inverse elements
		rec = np.abs(rec)
		# rec_nib = nib.Nifti1Image(rec, affine=np.eye(4))
		# nib.save(rec_nib, os.path.join('/home/rgu/Documents/ssm_results/left_femur', 'rec1.nii.gz'))
  

		return rec