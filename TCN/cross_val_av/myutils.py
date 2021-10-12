from scipy.io import loadmat
import sys
import torch
import numpy as np
import os
import random
from skimage.transform import rescale
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shutil
from PIL import Image, ImageDraw
import cv2
from torch.utils.data import Dataset


class FrameLevelDataset(Dataset):
	def __init__(self, x, labels):
		"""
			feats: Python list of numpy arrays that contain the sequence features.
				   Each element of this list is a numpy array of shape seq_length x feature_dimension
			labels: Python list that contains the label for each sequence (each label must be an integer)
		"""
		self.lengths = [sample.shape[1] for sample in x] # Find the lengths 

		self.x = self.zero_pad_and_stack(x)

		self.labels = self.zero_pad_and_stack(labels)

	def zero_pad_and_stack(self, x):
		"""
			This function performs zero padding on a list of features and forms them into a numpy 3D array
			returns
				padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
		"""
		
		maxLen = max(self.lengths)
		padded = np.zeros((len(x), x[0].shape[0], maxLen))
		padded = np.ones((len(x), x[0].shape[0], maxLen))
		for i, sample in enumerate(x):
			sequence_length = sample.shape[1]
			padded[i,:,:sequence_length] = sample 
			
		return padded

	def __getitem__(self, item):
		return self.x[item], self.labels[item], self.lengths[item]

	def __len__(self):
		return len(self.x)



def visualize_keypoints(feats, baf, path_to_data, vid_dir):
	vid_dir = 'VidSep_'+ vid_dir.split('.')[0]
	print(vid_dir)

	colors = ['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']
	outpath = '../../CheckFeatsPost/'+vid_dir.split('.')[0]
	try:
		shutil.rmtree(outpath)
	except:
		print("Error while deleting directory. Probably didn't exist")
	try:
		os.mkdir(outpath)
	except:
		print(outpath,' dir already exists')

	vidcap = cv2.VideoCapture(path_to_data+vid_dir+vid_dir+'.avi')
	print(path_to_data+vid_dir+'.avi')
	success, image = vidcap.read()
	h, w, layers = image.shape

	for count, frame in enumerate(feats.T):
		img = Image.new("RGB", (w, h)) 

		# DRAW LIMBS
		draw = ImageDraw.Draw(img)  
		draw.line([(frame[0], frame[1]), (frame[2], frame[3])], fill=colors[0], width=5) # 0-1
		draw.line([(frame[2], frame[3]), (frame[4], frame[5])], fill=colors[1], width=5) # 1-2
		draw.line([(frame[4], frame[5]), (frame[6], frame[7])], fill=colors[2], width=5) # 2-3
		draw.line([(frame[6], frame[7]), (frame[8], frame[9])], fill=colors[3], width=5) # 3-4
		draw.line([(frame[2], frame[3]), (frame[10], frame[11])], fill=colors[4], width=5) # 1-5
		draw.line([(frame[10], frame[11]), (frame[12], frame[13])], fill=colors[5], width=5) # 5-6
		draw.line([(frame[12], frame[13]), (frame[14], frame[15])], fill=colors[6], width=5) # 6-7
		draw.line([(frame[2], frame[3]), (frame[16], frame[17])], fill=colors[7], width=5) # 1-8
		draw.line([(frame[16], frame[17]), (frame[18], frame[19])], fill=colors[8], width=5) # 8-9
		draw.line([(frame[16], frame[17]), (frame[20], frame[21])], fill=colors[0], width=5) # 8-12

		img.save(outpath+'/'+outpath.split('/')[-1]+'_'+str(count).zfill(4)+'.jpg')
		del(draw)


def get_derivatives(feats):
	T = feats.T
	A = T[:-1,:] # NxM-1
	B = T[1:,:] # NxM-1
	A = (A - B) # coordinate shifts per frame (NxM-1)
	A = np.vstack((A,A[-1,:])) # NOTE: duplicate last column to ensure same size as features
	return A.T

def min_max_normalization(feats, maxX, maxY):
	N, M = feats.shape
	xs = np.array( [i for i in range(0, N, 2)] ) # even
	ys = np.array( [i for i in range(1, N+1, 2)] ) # odd	
	Xfeats = feats[xs,:]
	Yfeats = feats[ys,:]

	feats[xs,:] = feats[xs,:] / maxX
	feats[ys,:] = feats[ys,:] / maxY

	return feats

def get_average_std(feats):
	N, M = feats.shape
	xs = np.arange(0, N, 2)
	ys = np.arange(1, N, 2)

	sigmax = np.std(feats[xs], axis=0)

# TODO: change name and check CORRECTNESS
def midhip_normalize(feats, fp): # fp means fixed point

	N, M = feats.shape
	xs = np.arange(0, N, 2)
	ys = np.arange(1, N, 2)

	Cx = np.average(feats[xs], axis=0) # vector with dimensionality == duration
	Cy = np.average(feats[ys], axis=0)
	avrg_Cx = np.average(Cx) # value
	avrg_Cy = np.average(Cy)

	# NOTE: Standardize (Check correctness)
	sigmax = np.std(feats[xs], axis=0) # vector with dimensionality == duration
	sigmay = np.std(feats[ys], axis=0)
	avrg_sigmax = np.average(sigmax) # value
	avrg_sigmay = np.average(sigmay)

	feats[xs] = (feats[xs] - avrg_Cx) / avrg_sigmax 
	feats[ys] = (feats[ys] - avrg_Cy) / avrg_sigmay  
	
	return feats

# def data_generator(args, input_data, frames_to_rescale=None):
def data_generator(args, input_data):
	print('*********** Load '+input_data+' Data *************')
	# Training Data
	const_factor = 5
	X, Y, T, FNames = [], [], [], []
	train_path_to_data = args.project_dir+input_data+'/'

	if args.awgn and input_data=='Audio':
		train_path_to_data = args.project_dir+input_data+'SNR0/'

	for count, filename in enumerate(os.listdir(train_path_to_data)):

		# train_path_to_data = args.project_dir+input_data+'/'
		audnpzfile = np.load(args.project_dir+'Audio/'+filename)
		audbaf = audnpzfile['baf']
		n_audio_frames = audbaf.shape[0]

		npzfile = np.load(train_path_to_data+filename)
		feats = npzfile['feats'] # n_feats x time_duration
		baf = npzfile['baf']

		n_visual_frames = feats.shape[1] # video n_frames

		# ########################################################
		# if count%10==0 and args.visualize:
		# 	baf = npzfile['baf']
		# 	# path_to_data = '../../OpenPoseJsons/'
		# 	path_to_data = 'D:/users/grigoris/Datasets/SomeDataExtractedWithOpenPose/OpenPoseDataSep_toUse/'
		# 	visualize_keypoints(feats.T, baf, path_to_data, filename)
		# ########################################################
				
		# Normalize Skeleton Data
		if input_data=="HandROIs":
			baf = baf[:,0]
			# baf = rescale(baf, n_visual_frames/n_audio_frames)
			baf = np.array( [[y] for y in baf] )
			imgs = feats

		if input_data=="Hand":
			if args.rescaled:
				feats = rescale(feats, (1, n_audio_frames/n_visual_frames))  # upsample feats	
			feats = StandardScaler(with_std=False).fit_transform(feats.T).T


		# Normalize Skeleton Data
		if input_data=="Visual":
			# print('BBBBBBBBBB')		

			# NOTE: moved
			if args.rescaled:
				feats = rescale(feats, (1, n_audio_frames/n_visual_frames)) # upsample feats	

			if args.scaling=="standard": # TODO: probably need checking and correction
				if args.pca:
					feats = StandardScaler(with_std=False).fit_transform(feats.T).T 

					npzfileW = np.load(args.project_dir+'PCA_W_'+str(args.pca)+'.npz')
					W = npzfileW['W']
					X_v = feats.T
					feats = X_v.dot(W.T).T
				else:

					feats = StandardScaler(with_std=False).fit_transform(feats.T).T 

			elif args.scaling=="standard_alt":

				N, _ = feats.shape
				xs = np.arange(0, N, 2)
				ys = np.arange(1, N, 2)

				feats[xs] = StandardScaler().fit_transform(feats[xs])
				feats[ys] = StandardScaler().fit_transform(feats[ys])

			elif args.scaling=="fixed-midhip":

				# NOTE: NEW IDEA
				if args.pca: 
					npzfileW = np.load(args.project_dir+'PCA_W_'+str(args.pca)+'.npz')
					W = npzfileW['W']

					X_v = np.copy(feats.T) # NOTE: maybe need deepcopy?
					mu = np.average(X_v, axis=0) # shape 22
					s = np.std(X_v, axis=0) # shape 22	

					# Scale
					X_v = StandardScaler().fit_transform(X_v)
					# Transform
					pca_feats = X_v.dot(W.T).T
					
				midhip_mean = (np.mean(feats[16,:]), np.mean(feats[17,:]))
				feats = midhip_normalize(feats, midhip_mean)

		# ########################################################
		# if count%10==0 and args.visualize:
		# 	# path_to_data = '../../OpenPoseJsons/'
		# 	path_to_data = 'D:/users/grigoris/Datasets/SomeDataExtractedWithOpenPose/OpenPoseDataSep_toUse/'
		# 	visualize_keypoints(feats.T, np.array([]), path_to_data, filename)			
		# ########################################################

		# ADD DERIVATIVES
		if input_data=="Visual" or input_data=="Hand":
			vel = get_derivatives(feats)
			feats = np.concatenate((feats, vel), axis=0)
			accel = get_derivatives(vel)
			feats = np.concatenate((feats, accel), axis=0)
		if args.pca and input_data=="Visual":
			vel = get_derivatives(pca_feats)
			pca_feats = np.concatenate((pca_feats, vel), axis=0)
			accel = get_derivatives(vel)
			pca_feats = np.concatenate((pca_feats, accel), axis=0)
		# NOTE: NEW IDEA
		if args.pca and input_data=="Visual" and args.scaling=='fixed-midhip': 
			feats = np.concatenate((feats, pca_feats), axis=0)	

		if args.deriv and input_data=="Audio":
			vel = get_derivatives(feats)
			feats = np.concatenate((feats, vel), axis=0)		

		# output smoothing
		locations = np.nonzero(baf)[0]

		baf[locations] = 1
		try:
			baf[locations+1] = 1
			baf[locations-1] = 1
			if args.rescaled or args.modality=='Audio':
				baf[locations+2] = 1
				baf[locations-2] = 1
				if args.instr!='guitar':
					baf[locations+3] = 1
					baf[locations-3] = 1		
					baf[locations+4] = 1
					baf[locations-4] = 1

		except IndexError as e:
			print('MyWarning: exact frame ', str(locations[-1,]), ', ', e)
			pass

		baf_ = 1.0 - baf
		baf = np.concatenate((baf, baf_), axis=1)

		if input_data=='HandROIs':
			X.append(imgs)		
		else:
			X.append(feats.T) #NOTE: inverse
		Y.append(baf)
		FNames.append(filename.split('.')[0])

		# Groundtruth
		times = npzfile["onset_times"]
		T.append(times)

	# Shuffle Data
	idx_list = np.arange(len(X), dtype="int32")
	np.random.seed(args.seed)
	np.random.shuffle(idx_list)
	X = [X[i] for i in idx_list]
	Y = [Y[i] for i in idx_list]
	T = [T[i] for i in idx_list]
	FNames = [FNames[i] for i in idx_list]


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


	for data in [X, Y]:
		for i in range(len(data)):
			data[i] = torch.Tensor(data[i].astype(np.float64))
			
	# Create Folds
	N=args.n_folds
	XFold=[]
	YFold=[]
	TFold=[]
	FFold=[]
	p = len(X)*(1/N)
	for i in range(N-1):

		if i>1 and args.augment:
			XX, YY, TT, FF = augment(X[round(i*p):round((i+1)*p)], Y[round(i*p):round((i+1)*p)], T[round(i*p):round((i+1)*p)], FNames[round(i*p):round((i+1)*p)])
			print('Fold', i, 'indices:', round(i*p), round((i+1)*p))
			XFold.append(XX)
			YFold.append(YY)
			TFold.append(TT)
			FFold.append(FF)	
		else:
			print('Fold', i, 'indices:', round(i*p), round((i+1)*p))
			XFold.append(X[round(i*p):round((i+1)*p)])
			YFold.append(Y[round(i*p):round((i+1)*p)])
			TFold.append(T[round(i*p):round((i+1)*p)])
			FFold.append(FNames[round(i*p):round((i+1)*p)])

	print('Fold', N-1, 'indices:', round((N-1)*p), len(X))
	XFold.append(X[round((N-1)*p):])
	YFold.append(Y[round((N-1)*p):])
	TFold.append(T[round((N-1)*p):])
	FFold.append(FNames[round((N-1)*p):])

	
	# Check Algorithm Correctness
	for i in range(args.n_folds):
		for j in range(args.n_folds):
			if i==j: continue
			isdisjoint = set(FFold[i]).isdisjoint(set(FFold[j]))
			if not isdisjoint:
				print('There is at least one couple of folds which are non-disjoint!!')
				exit(1)

	del X, Y

	return XFold, YFold, TFold, FFold


def augment(X_train, Y_train, T_train, F_train):

	extraX = []
	extraY = []
	extraT = []
	extraF = []

	print(len(X_train))
	for video, y, t, name in zip(X_train, Y_train, T_train, F_train):
		if name.split('_')[1] in ['vn', 'va']:
			plt.imshow(video[100, :, :, :].numpy())
			plt.show()
			new_video = video.transpose(1, 2).flip(1) # h, w, 3 ->  w, h, 3
			print(new_video.size())
			new_video = torch.flip(new_video, [1])
			plt.imshow(new_video[100, :, :, :].numpy())
			extraX.append(new_video)
			extraY.append(y)
			extraT.append(t)
			extraF.append(name)

	X_train = X_train + extraX
	Y_train = Y_train + extraY
	T_train = T_train + extraT
	F_train = F_train + extraF
	print(len(X_train))

	return X_train, Y_train, T_train, F_train

def fold_generator(args,  XFold, YFold, TFold, FFold):

	N=args.n_folds
	for fold_id in range(args.n_folds):

		X_test = XFold[fold_id] 	# e.g. fold_id = 0
		Y_test = YFold[fold_id]
		T_test = TFold[fold_id]
		F_test = FFold[fold_id]

		X_valid = XFold[(fold_id+1)%N]	# e.g. fold_id%N+1 = 1
		Y_valid = YFold[(fold_id+1)%N]
		T_valid = TFold[(fold_id+1)%N]
		F_valid = FFold[(fold_id+1)%N]

		X_train=[]
		Y_train=[]
		T_train=[]
		F_train=[]
		for i in range(N-2):
			X_train+=XFold[(fold_id+i+2)%N]
			Y_train+=YFold[(fold_id+i+2)%N]
			T_train+=TFold[(fold_id+i+2)%N]
			F_train+=FFold[(fold_id+i+2)%N]

		yield X_train, X_valid, X_test, Y_train, Y_valid, Y_test, T_train, T_valid, T_test, F_train, F_valid, F_test

if __name__ == "__main__":
	input_data = 'Visual'
	X_train, X_test = data_generator(input_data)
	print(type(X_train[0]))