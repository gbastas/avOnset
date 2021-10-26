from scipy.io import loadmat
import sys
import torch
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shutil
from torch.utils.data import Dataset


def data_generator(args, input_data):
	print('*********** Load '+input_data+' Data *************')
	# Training Data
	const_factor = 5
	X, Y, T, FNames = [], [], [], []
	train_path_to_data = args.project_dir+'Train/'
	if args.awgn and input_data=='Audio':
		train_path_to_data = args.project_dir+input_data+'SNR0/'

	for count, filename in enumerate(os.listdir(train_path_to_data)):

		# if 'solo' in filename:

		npzfile = np.load(train_path_to_data+filename)

		otimes = npzfile['onset_times']
		feats = npzfile['feats'] # n_feats x time_duration
		baf = npzfile['baf']

		n_audio_frames = baf.shape[0]
		n_visual_frames = feats.shape[1] # video n_frames
		
		T.append(otimes)

		if args.deriv and input_data=="Audio":
			vel = get_derivatives(feats)
			feats = np.concatenate((feats, vel), axis=0)		

		locations = np.nonzero(baf)[0]

		# print('AAAA', filename)
		# print(locations[-1])
		# print(otimes[-1])

		# TODO: exepermient on baf output
		# [baf, p] = square_baf_const(baf, const_factor, v=0.7) # NOTE: smooth
		# output smoothing
		baf[locations] = 1
		try:
			baf[locations+1] = 1
			baf[locations-1] = 1
			baf[locations+2] = 1
			baf[locations-2] = 1

		except IndexError as e:
			print('MyWarning: exact frame ' + str(locations[-1,]) + ', ' + str(e))
			pass

		baf_ = 1.0 - baf
		baf = np.concatenate((baf, baf_), axis=1)

		X.append(feats.T) #NOTE: inverse
		Y.append(baf)
		FNames.append(filename.split('.')[0])


	# create shuffled validation and training sets NOTE: Should I shuffle?
	train_idx_list = np.arange(len(X), dtype="int32")
	np.random.seed(0)
	np.random.shuffle(train_idx_list)
	print(train_idx_list[-10:])
	X_train = [X[i] for i in train_idx_list[:-10]]
	Y_train = [Y[i] for i in train_idx_list[:-10]]
	T_train = [T[i] for i in train_idx_list[:-10]]
	F_train = [FNames[i] for i in train_idx_list[:-10]]
	X_valid = [X[i] for i in train_idx_list[-10:]]
	Y_valid = [Y[i] for i in train_idx_list[-10:]]	
	T_valid = [T[i] for i in train_idx_list[-10:]]
	F_valid = [FNames[i] for i in train_idx_list[-10:]]
	
	###### Test Data ############
	def checkforDuplicates(listOfElems):
		for idx, elem in enumerate(listOfElems):
			if listOfElems.count(elem) > 1:
				print(idx, elem)
				return True
		return False

	# Check Algorithm Correctness
	if checkforDuplicates(FNames):
		print('Duplicates appear in the dataset!!' )
		exit(1)

	X_test, Y_test, T_test, F_test = [], [], [], []
	test_path_to_data = args.project_dir+'Test/'#+input_data+'/'

	for filename in os.listdir(test_path_to_data):
		npzfile = np.load(test_path_to_data+filename)
		feats = npzfile['feats'].T

		# if args.deriv:
		# 	vel = get_derivatives(feats)
		# 	feats = np.concatenate((feats, vel), axis=1)

		baf = npzfile['baf']

		# [baf, p] = square_baf_const(baf, const_factor)

		locations = np.nonzero(baf)[0]

		baf[locations] = 1
		try:
			baf[locations+1] = 1
			baf[locations-1] = 1
			baf[locations+2] = 1
			baf[locations-2] = 1
		except IndexError as e:
			print('MyWarning: exact frame ' + str(locations[-1,]) + ', ' + str(e))
			pass

		baf_ = 1.0 - baf
		baf = np.concatenate((baf, baf_), axis=1)

		X_test.append(feats) #NOTE: inverse
		Y_test.append(baf)
		F_test.append(filename.split('.')[0])

		# Groundtruth
		times = npzfile['onset_times']
		T_test.append(times)

	# From (2D) numpy arrays ---to---> torch tensors 
	for data in [X_train, X_valid, X_test, Y_train, Y_valid, Y_test, T_train, T_valid, T_test]:
		for i in range(len(data)):
			data[i] = torch.Tensor(data[i].astype(np.float64))

	return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), (T_train, T_valid, T_test), (F_train, F_valid, F_test)

def fold_generator(args, XFold, YFold, TFold, FFold): # pseudfolds

	(X_train, X_valid, X_test) = XFold
	(Y_train, Y_valid, Y_test) = YFold
	(T_train, T_valid, T_test) = TFold
	(F_train, F_valid, F_test) = FFold

	yield X_train, X_valid, X_test, Y_train, Y_valid, Y_test, T_train, T_valid, T_test, F_train, F_valid, F_test