'''
This is a script for feature extraction of the URMP Dataset and their subesequent storage to npz files.
It parses the skeleton data to retrieve visual information 
and then the actual performance directory to exctract audio information from wav 
or extra visual information from the monophonic performances (mkv files).
'''


import numpy as np
import argparse
import sys
import os
import shutil
import json
import matplotlib.pyplot as plt
import cv2
import librosa
from PIL import Image, ImageDraw
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import util
from shutil import copyfile
sys.path.append("../")
import copy 
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from tools.beats import smooth_baf, square_baf_const
# np.set_printoptions(threshold=sys.maxsize)
import math 

FS = 48000
# HOP = 512 # 10 ms
HOP = 480 # 10 ms
W_SIZE = 2048 # 42.7 ms

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def get_salience(Z): # Z == feats.T (samples x features)
	n = 150 # past frames
	k = Z.shape[1]
	pca = PCA(n_components=k)
	S = []
	Cmp1 = []
	for t in range(n, Z.shape[0]):
		pca.fit(Z[t-n:t, :])
		V = pca.components_
		cmp1 = V[0,:]
		s = abs( np.dot(Z[t, :], cmp1) )
		S.append(s)
		Cmp1.append(cmp1)

	Salience = np.array(S)
	Cmp1 = np.array(Cmp1)
	S = np.array(S)

	return S, Cmp1

def get_derivatives(feats):
	A = feats[:-1,:] # NxM-1
	B = feats[1:,:] # NxM-1
	A = (A - B) # coordinate shifts per frame (NxM-1)
	A = np.vstack((A,A[-1,:])) # duplicate lsat column to ensure same size as features
	return A

def scan_folder_sep(parent):

	KeyPoints=np.array( [np.zeros(75)] )
	leftHandKeyPoints=np.array( [np.zeros(63)] )
	rightHandKeyPoints=np.array( [np.zeros(63)] )
	# now extract 75-dimensional vector for each person and each frame
	for file_name in sorted(os.listdir(parent)):
		if file_name.endswith(".json"):
			directory=parent+'/'+file_name
			with open(directory) as f:
				data=json.load(f)

			people=data['people']
			if len(people)==0:
				print("no skeleton appeared")
				keypoints=np.array( np.zeros(75) )
			else:
				# body
				keypoints=np.array(people[0]['pose_keypoints_2d'])
				# left hand
				hand_left_keypoints_2d = np.array(people[0]['hand_left_keypoints_2d'])
				# right hand
				hand_right_keypoints_2d = np.array(people[0]['hand_right_keypoints_2d'])			# stack even when no skeleton is captured	
			KeyPoints = np.vstack((KeyPoints, keypoints))
			leftHandKeyPoints = np.vstack((leftHandKeyPoints, hand_left_keypoints_2d))
			rightHandKeyPoints = np.vstack((rightHandKeyPoints, hand_right_keypoints_2d))

	KeyPoints = np.delete(KeyPoints, 0, axis=0) # delete the very first row comprised of zeroes
	leftHandKeyPoints = np.delete(leftHandKeyPoints, 0, axis=0) # delete the very first row comprised of zeroes
	rightHandKeyPoints = np.delete(rightHandKeyPoints, 0, axis=0) # delete the very first row comprised of zeroes

	return KeyPoints, leftHandKeyPoints, rightHandKeyPoints


def interpolate_keypoints(A):
	N, M = A.shape
	for i in range(N):
		flag = False
		# Handle unfound zero-points from the very first frame
		s = np.argmax(A[i]>0)
		for t in range(s): 
			A[i,t] = A[i,s]
		# Handle unfound zero-points from the last frame, from left to right
		e = np.argmax(A[i,::-1]>0)
		for t in range(M-1, M-e-1, -1):
			A[i,t] = A[i,M-e-1]
		# Interpolate when zero points are found
		for t in range(s+1,M-e): # 
			if A[i,t]==0:
				if not(flag):
					k = t-1
					l = t+1
					flag = True
				else:
					l += 1
			else:
				if flag:
					A[i, k:l] = np.linspace(A[i,k], A[i,l], l-k, endpoint=False)
					flag=False
	return A


def hand_center_mass(A):
	N, M = A.shape
	xs = np.array( [i for i in range(0, N, 2)] ) # even
	ys = np.array( [i for i in range(1, N+1, 2)] ) # odd
	X = A[xs,:]
	Y = A[ys,:]

	# Check sth for correctness of method
	n_noncommon_zero_elements = np.count_nonzero(( (X[X==0] == Y[Y==0]) == False ) ) 
	if n_noncommon_zero_elements > 0:
		print('There were', n_noncommon_zero_elements, 'zero_points in different positions for x and y axis. Need to reconsider processing method.')
		exit(1)

	# Find center mass for every time frame
	mu = []
	for j in range(M): # loop through frames
		x = X[:, j]
		non_zero_els = x[x>0]
		if len(non_zero_els)>0:
			x_mu = sum(non_zero_els)/len(non_zero_els)
		else:
			x_mu = 0 # eliminate to further interpolate

		y = Y[:, j]
		non_zero_els = y[y>0]
		if len(non_zero_els)>0:
			y_mu = sum(non_zero_els)/len(non_zero_els)
		else:
			y_mu = 0 # eliminate to further interpolate

		mu.append([x_mu, y_mu])

	return np.array(mu).T



def eliminate_abrupt_keypoint_shifts(A):
	N, M = A.shape
	xs = np.array( [i for i in range(0, N, 2)] ) # even
	ys = np.array( [i for i in range(1, N+1, 2)] ) # odd	
	X = A[xs,:]
	Y = A[ys,:]
	prev_corr = [0] * (N//2)
	for i in range(N//2):
		# Find first non-zero element
		sx = np.argmax(X[i]>0)
		sy = np.argmax(Y[i]>0)
		s = max(sx, sy)
		prev_corr[i]=s

		for t in range(s+1,M):
			prev_corr[i] = t-1 # NOTE: try this method for now

			mrgn = 0.10 * ( (X[0,t]-X[8,t])**2 + (Y[0,t]-Y[8,t])**2 )**0.5 # head-hip distanaces^2 (maximal regular movement in a ≈30-FPS video)
			# mrgn = 25.5 # make adapatable! ( head-hip distanaces^2 (maximal regular movement in a ≈30-FPS video) )
			shft = ( (X[i, prev_corr[i]] - X[i,t])**2 + (Y[i,prev_corr[i]] - Y[i,t])**2 )**0.5

			if shft > mrgn and (X[i,prev_corr[i]]!=0 and Y[i,prev_corr[i]]!=0):
				A[2*i,t] = 0
				A[2*i+1,t] = 0

	return A

def centered_moving_average(a, n=5) :
	ret = np.cumsum(a, dtype=float, axis=1)
	ret[:,n:] = (ret[:,n:] - ret[:,:-n])/n
	for i in range(1,n):
		ret[:,i] = ret[:,0]
	return ret


def eliminate_low_confidence_keypoints(conf, feats):
	Ii, It =  np.where(conf<0.2)

	Iix=2*Ii
	Iiy=2*Ii+1
	feats[Iix, It]=0
	feats[Iiy, It]=0

	return feats

def visualize_hand_joints(rois, hand_feats, hand_feats_mu, baf, path_to_data, vid_dir, px):
	colors = ['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']
	outpath = '../CheckHandFeats/'+vid_dir
		
	try:
		os.mkdir('../CheckHandFeats/')
	except: 
		print('../CheckHandFeats/ dir already exists')

	example_img = rois[0,:,:,:]
	h, w, layers = example_img.shape

	if baf.shape[0]>0:
		[baf, p] = smooth_baf(baf, 3) # NOTE: smooth
		baf_ = 1.0 - baf
		baf = np.concatenate((baf, baf_), axis=1)
		baf = np.swapaxes(baf,1,0)[0]

	os.mkdir(outpath)
	for i, frame in enumerate(hand_feats.T):
		if baf.shape[0]>0:
			shade = (int(baf[i]*255), int(baf[i]*255), int(baf[i]*255))
			img = Image.new("RGB", (w, h), shade) 
		else:
			img = Image.new("RGB", (w, h)) 

		roi = rois[i,:,:,:]
		# DRAW LIMBS
		draw = ImageDraw.Draw(img)   

		# New Origin
		cm = hand_feats_mu[:,i]
		joint = []
		for j in range(0,42,2):
			x_ = frame[j] - (cm[0]-px//2) # NOTE if 0 is correct
			y_ = frame[j+1] - (cm[1]-px//2)
			joint += [(x_, y_)]

		######## Classic w, h representation ########

		draw.line([joint[0], joint[1]], fill=colors[1], width=2) #
		draw.line([joint[1], joint[2]], fill=colors[2], width=2) #
		draw.line([joint[2], joint[3]], fill=colors[3], width=2) # 
		draw.line([joint[3], joint[4]], fill=colors[4], width=2) # 
		draw.line([joint[0], joint[5]], fill=colors[5], width=2) # 
		draw.line([joint[5], joint[6]], fill=colors[6], width=2) # 
		draw.line([joint[6], joint[7]], fill=colors[6], width=2) # 
		draw.line([joint[7], joint[8]], fill=colors[7], width=2) # 
		draw.line([joint[0], joint[9]], fill=colors[8], width=2) # 
		draw.line([joint[9], joint[10]], fill=colors[0], width=2) # 
		draw.line([joint[10], joint[11]], fill=colors[1], width=2) # 
		draw.line([joint[11], joint[12]], fill=colors[2], width=2) # 
		draw.line([joint[0], joint[13]], fill=colors[3], width=2) # 
		draw.line([joint[13], joint[14]], fill=colors[4], width=2) # 
		draw.line([joint[14], joint[15]], fill=colors[5], width=2) # 
		draw.line([joint[15], joint[16]], fill=colors[6], width=2) # 
		draw.line([joint[0], joint[17]], fill=colors[7], width=2) # 
		draw.line([joint[17], joint[18]], fill=colors[8], width=2) # 
		draw.line([joint[18], joint[19]], fill=colors[0], width=2) # 
		draw.line([joint[19], joint[20]], fill=colors[1], width=2) # 

		toshow = roi + img

		plt.imsave(outpath+'/'+vid_dir+'_'+str(i).zfill(4)+'.jpg', toshow)

	
def visualize_keypoints(feats, baf, path_to_data, vid_dir):
	colors = ['#37AB65', '#3DF735', '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']

	vidcap = cv2.VideoCapture(path_to_data+vid_dir+'.avi')
	# print(path_to_data)
	success, framecap = vidcap.read()
	# print(framecap)
	h, w, layers = framecap.shape

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter('video.avi', fourcc, 20.0, (w ,h))

	print()

	for count, frame in enumerate(feats.T):
		if baf.shape[0]>0:
			shade = (int(baf[count]*255), int(baf[count]*255), int(baf[count]*255))
			img = Image.new("RGB", (w, h), shade) 
		else:
			img = Image.new("RGB", (w, h)) 

		# DRAW LIMBS
		draw = ImageDraw.Draw(img) 

		draw.line([(frame[2], frame[3]), (frame[16], frame[17])], fill=colors[7], width=5) # 1-8
		draw.line([(frame[0], frame[1]), (frame[2], frame[3])], fill=colors[0], width=5) # 0-1
		draw.line([(frame[2], frame[3]), (frame[4], frame[5])], fill=colors[1], width=5) # 1-2
		draw.line([(frame[4], frame[5]), (frame[6], frame[7])], fill=colors[2], width=5) # 2-3
		draw.line([(frame[6], frame[7]), (frame[8], frame[9])], fill=colors[3], width=5) # 3-4
		draw.line([(frame[2], frame[3]), (frame[10], frame[11])], fill=colors[4], width=5) # 1-5
		draw.line([(frame[10], frame[11]), (frame[12], frame[13])], fill=colors[5], width=5) # 5-6
		draw.line([(frame[12], frame[13]), (frame[14], frame[15])], fill=colors[6], width=5) # 6-7
		# draw.line([(frame[2], frame[3]), (frame[16], frame[17])], fill=colors[7], width=5) # 1-8
		draw.line([(frame[16], frame[17]), (frame[18], frame[19])], fill=colors[8], width=5) # 8-9
		draw.line([(frame[16], frame[17]), (frame[20], frame[21])], fill=colors[0], width=5) # 8-12

		try: # if hand (i.e. center mass) of hand keypoints is extracted...
			draw.line([(frame[14], frame[15]), (frame[22], frame[23])], fill=colors[7], width=5) # 7-palm
		except:
			print("Couldn't print hand center mass")
			pass
		
		image = np.array(img)

		video.write(image)
		ret, framecap = vidcap.read()

	video.release()

def get_roi(path_to_mkv_file, hand_feats_mu, px):
	vidcap = cv2.VideoCapture(path_to_mkv_file)
	t=0
	success, image = vidcap.read()
	h, w, layers = image.shape
	ROIs=[]
	while success:
		cm = hand_feats_mu[:,t] 
		x, y = int(cm[0]), int(cm[1])
		roi = image[-px//2+y:px//2+y, -px//2+x:px//2+x]
		
		ROIs.append(roi)

		t +=1 # NOTE: check again
		success, image = vidcap.read()

	print(len(ROIs), ROIs[0].shape)


	if min([ROIs[i].shape[0] for i in range(len(ROIs))]) != px:
		print('PROBLEM!!!!!!!!!!!!!!')
		exit(1)
		rois = np.array(ROIs)

	if min([ROIs[i].shape[1] for i in range(len(ROIs))]) == px:
		rois = np.array(ROIs)
	else:
		# NOTE: only VidSep_1_vn_44_K515.mkv has this problem
		print('\nHandling non-100x100 images!!\n')
		padded_ROIs = []
		for roi in ROIs:
			# Ηere handling only width on the right side (i.e. reuse of the last pixel-column, copy & append)
			w = roi.shape[1]
			
			if w == 0:
				roi = np.copy(padded_ROIs[-1])
			elif w<px:
				for j in range(w,px):
					last_roi_column = np.zeros(roi[:,j-1,:].shape, dtype='uint8').reshape(px,1,3)
					roi = np.append(roi, last_roi_column, axis=1) # j-1 is th

			padded_ROIs.append(roi)

		rois = np.array(padded_ROIs)
	return rois


def get_opt_flow(rois):
	print(rois.shape)
	outs = []
	rois_cv = rois[:,:,:,:]
	frame1 = rois_cv[0,:,:,:]
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	for i in range(1, len(rois_cv)):
		frame2 = rois_cv[i,:,:]
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		######### Optical Flow ##########
		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		#################################
		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		
		# NOTE: uncomment for visualization
		# cv2.imshow('frame', frame2)
		# cv2.imshow('of', rgb)
		outs.append(rgb)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			# cv2.imwrite('opticalfb.png',frame2)
			cv2.imwrite('opticalhsv.png',rgb)
			outs.append(rgb)
		prvs = next

	return np.array(outs)


def prepare_data(args):

	# Create path_ot_store
	path_to_store = args.pathToData+args.method
	if args.log:
		path_to_store += 'log'
	if args.spectralflux:
		path_to_store += 'spectralflux'
	if args.mov_avg:
		path_to_store += 'mov_avg'
	path_to_store += '/'

	if not args.visualize:
		try:
			shutil.rmtree(path_to_store)
		except:
			print('Error while deleting directory')

		os.mkdir(path_to_store)
		os.mkdir(path_to_store+'Visual')
		os.mkdir(path_to_store+'Audio')
		os.mkdir(path_to_store+'HandROIs')
		os.mkdir(path_to_store+'Hand')
		os.mkdir(path_to_store+'Li')

	# TODO: fix paths
	# windows 
	path_to_data = 'D:/users/grigoris/Datasets/OpenPoseExtractedData/OpenPoseData_all_hand/'
	# linux
	path_to_data = "/media/gbastas/New Volume/users/grigoris/Datasets/OpenPoseExtractedData/OpenPoseData_all_hand/"
	# path_to_data = 'C:/Users/g.bastas/openpose/OpenPoseData/'

	path_to_copy_data_AuSep = '../wav/'
	path_to_copy_data_Notes = '../annotations/'
	# get valid subset of keypoints coordinates (format: [x0, y0, ... , x9, y9, x12, y12])
	subset_pos = np.array( [ [3*i, 3*i+1] for i in range(0,10) ] + [ [3*12, 3*12+1] ] ) # keep points: 0-9 & 12 
	subset_pos = subset_pos.flatten()
	# the same for hand keypoints
	hand_pos = np.array( np.array( [ [3*i, 3*i+1] for i in range(0,21) ] ) )
	hand_pos = hand_pos.flatten()
	# get valid subset of keypoints confidence
	confidence_pos = np.array( [ [3*i+2] for i in range(0,10) ] + [ [3*12+2] ] ) # keep points: 0-9 & 12
	confidence_pos = confidence_pos.flatten()
	# the same for hand keypoints
	hand_conf_pos = np.array( [3*i+2 for i in range(0,21)] )

	count=0

	flag=True
	Z_all = []
	for subdir in sorted(os.listdir(path_to_data)): # e.g. subdir=='VidSep_1_cl_19_Pavane' that includes json files 
		try:
			# Keep only string instruments
			if args.strings and (subdir.split("_")[2] not in ['vn','vc','va','db']): 
				continue

			if args.no_strings and (subdir.split("_")[2] in ['vn','vc','va','db']): 
				continue

			# Avoid parsing the stored skeleton-videos. Focus on .json files.
			if subdir.startswith('.') or subdir.endswith('.avi'):
				continue

			# Get visual features i.e. 10 (very specific) skeleton keypoints
			keypoints, leftHandKeyPoints, rightHandKeyPoints = scan_folder_sep(path_to_data + subdir)
			# body
			visual_feats = keypoints[:,subset_pos].T
			conf = keypoints[:,confidence_pos].T
			# (left) hand
			hand_feats = leftHandKeyPoints[:,hand_pos].T
			hand_conf = leftHandKeyPoints[:, hand_conf_pos].T
			# right hand
			hand_r_feats = rightHandKeyPoints[:,hand_pos].T
			hand_r_conf = rightHandKeyPoints[:, hand_conf_pos].T
			# # merge two hands 
			# hand_feats = np.vstack((hand_feats, hand_r_feats))
			# hand_conf = np.vstack((hand_conf, hand_r_conf))


			###### FILTER OUT #######
			visual_feats = eliminate_abrupt_keypoint_shifts(visual_feats)
			visual_feats = eliminate_low_confidence_keypoints(conf, visual_feats) # discard points with confidence value  <0.2
			visual_feats = interpolate_keypoints(visual_feats)

			hand_feats = eliminate_abrupt_keypoint_shifts(hand_feats)
			hand_feats = eliminate_low_confidence_keypoints(hand_conf, hand_feats)

			hand_feats_mu = hand_center_mass(hand_feats)
			hand_feats_mu = interpolate_keypoints(hand_feats_mu)

			hand_feats = interpolate_keypoints(hand_feats)

			##### Moving Average ####
			if args.mov_avg:
				visual_feats = centered_moving_average(visual_feats, n=5)

				hand_feats_mu = np.array([np.mean(hand_feats_mu, axis=1) for i in range(hand_feats_mu.shape[1])] ).T # (seq_len x 2)

				print("hand_feats_mu", hand_feats_mu.shape, hand_feats.shape)

				hand_feats = centered_moving_average(hand_feats, n=3)

			Z = get_derivatives(visual_feats).T # speed martix
			Z_all.append(Z)

			# Check for zero-point elimination
			if (visual_feats==0).sum() > 0:
				print("Filtering didn't go as it shoud!")
				exit(1)
			else:
				print("Nicely Filtered!")

			# Get onset_times
			dir_keyword = '_'.join(subdir.split('_')[-2:]) # e.g. 03_Dbance
			perfID = subdir.split('_',2)[1] # e.g. '1_fl_03_Dance'

			# Find the proper URMP dir in order to get the right files containing onset locations, TODO: fix paths
			# windows
			urmp_path = 'D:/users/grigoris/Datasets/uc3/Dataset/'
			# linux
			urmp_path = "/media/gbastas/New Volume/users/grigoris/Datasets/uc3/Dataset/"
			for find_dir in sorted(os.listdir(urmp_path)):
				if not(find_dir.startswith('.')) and (dir_keyword in find_dir):
					uc3_subdir = find_dir
					break
			print(uc3_subdir) # e.g. 03_Dance_fl_cl

			# Once the proper dir is found we can parse the files containing onset_times and the audio recordings 
			# found in dirs containing all solo recordings of each polyphonic performance.
			path_to_uc3_subdir = urmp_path+uc3_subdir+'/'
			c=0
			for count, filename in enumerate(sorted(os.listdir(path_to_uc3_subdir))):

				if filename.startswith('Notes_'+perfID): # Get files containing onset_times
					print('Target:', path_to_uc3_subdir+filename)
					path_to_Notes_file = path_to_uc3_subdir+filename
					onset_times = np.loadtxt(path_to_Notes_file, usecols=0)

				# Extract Audio Features of the Corresponding Performance
				elif filename.startswith('AuSep_'+perfID):
					path_to_wav_file = path_to_uc3_subdir+filename

					audio_data_mix, sr = librosa.load(path_to_wav_file, sr=args.fs)

					####### ADD NOISE ######### 
					if args.awgn:
						SNRdb=-10
						audio_data_mix = add_awgn(audio_data_mix, SNRdb=SNRdb)
						if flag:
							print('Saving audio!!!!')
							flag=False
							librosa.output.write_wav('signal_noise_out_'+str(SNRdb)+'.wav', audio_data_mix, sr) 	# save sample

					# Extract audio features of our choice using librosa
					if args.method=='chromas':
						audio_feats = librosa.feature.chroma_stft(audio_data_mix, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
					elif args.method=='chromas_norm':
						audio_feats = librosa.feature.chroma_stft(audio_data_mix, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
						audio_feats = librosa.util.normalize(audio_feats, norm=2., axis=1)
					elif args.method=='melspec':
						audio_feats = librosa.feature.melspectrogram(audio_data_mix, sr=args.fs, n_mels=40, n_fft=args.w_size, hop_length=args.hop)
					# Compute log-power
					if args.log:
						audio_feats = librosa.power_to_db(audio_feats, ref=np.max)
					# Compute spectralflux
					if args.spectralflux:
						onset_envelope = np.array([librosa.onset.onset_strength(audio_data_mix, sr=args.fs)])
						audio_feats = np.concatenate((audio_feats, onset_envelope), axis=0)

				# About the visual component, if we want pixel information instead of just skeletons we need to parse the right monophonic performance
				elif filename.startswith('VidSep_'+perfID) and filename.endswith('.mkv'):
					c+=1
					print(filename)
					path_to_mkv_file = path_to_uc3_subdir+filename
					if args.HandROIs:
						if args.strings and (filename.split("_")[2] in ['vn','va']): 
							px=100
						if args.strings and (filename.split("_")[2] in ['vc','db']): 
							px=200
						hand_rois = get_roi(path_to_mkv_file, hand_feats_mu, px=px) # (duration, 100, 100, 3)		
						if px==200:
							hand_rois = [util.img_as_ubyte(resize(img, (100,100,3))) for img in hand_rois] # rescaled and uint8 
						hand_rois = np.array(hand_rois)

						if args.optflow:
							print('**** OPTICAL FLOW ****')
							hand_rois = get_opt_flow(hand_rois)

			n_audio_frames = audio_feats.shape[1]
			n_visual_frames = visual_feats.shape[1]


			# Visualize ROIs
			if args.visualize:
				# hands
				visualize_hand_joints(hand_rois, hand_feats, hand_feats_mu, np.array([]), path_to_data, subdir)
				# body
				print('**** VISUALIZING ****')
				visual_feats = np.vstack((visual_feats, hand_feats_mu))
				# ##################################################
				visualize_keypoints(visual_feats, np.array([]), path_to_data, subdir)
				# ##################################################
				continue

			# STORE
			if not args.visualize:

				onset_frames = librosa.core.time_to_samples(onset_times, sr=29.97) #NOTE
				if not args.visualize:
					pixbaf = np.array( [np.zeros( hand_rois.shape[0] )])
					pixbaf[0, onset_frames] = 1.
					pixbaf = np.swapaxes(pixbaf, 0, 1)
				
				# Create baf (e.g. [[0.],[0.],[1.],[0.],...]) array
				onset_frames = librosa.core.time_to_frames(onset_times, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
				baf = np.array( [np.zeros( audio_feats.shape[1] )])
				baf[0, onset_frames] = 1.
				baf = np.swapaxes(baf, 0, 1)

				filename = subdir.split('_',1)[-1]
				# np.savez(path_to_store+'Visual/'+filename+'.npz', feats=visual_feats, baf=baf, onset_times=onset_times)
				np.savez(path_to_store+'Visual/'+filename+'.npz', feats=visual_feats, baf=pixbaf, onset_times=onset_times)
				npzfile = np.load(path_to_store+'Visual/'+filename+'.npz') 
				print('*****SKELETONS*****')
				print('feats:', npzfile['feats'].shape) # (n_feats x n_samples)
				print('baf:', npzfile['baf'].shape) 
				print('onset_times:', npzfile['onset_times'].shape) # 

				np.savez(path_to_store+'Hand/'+filename+'.npz', feats=hand_feats, baf=baf, onset_times=onset_times)
				npzfile = np.load(path_to_store+'Hand/'+filename+'.npz')
				print('*****HANDS*****')
				print('feats:', npzfile['feats'].shape) # (n_feats x n_samples)
				print('baf:', npzfile['baf'].shape) 
				print('onset_times:', npzfile['onset_times'].shape) # 


			if args.HandROIs:
					np.savez(path_to_store+'HandROIs/'+filename+'.npz', feats=hand_rois, baf=pixbaf, onset_times=onset_times)
					# np.savez_compressed(path_to_store+'HandROIs/'+filename+'.npz', feats=np.int8(hand_rois), baf=pixbaf, onset_times=onset_times)
					npzfile = np.load(path_to_store+'HandROIs/'+filename+'.npz')
					print('*****HANDROIs*****')
					print('feats:', npzfile['feats'].shape) # (n_feats x n_samples)
					print('baf:', npzfile['baf'].shape) 
					print('onset_times:', npzfile['onset_times'].shape) # 

				np.savez(path_to_store+'Audio/'+filename+'.npz', feats=audio_feats, baf=baf, onset_times=onset_times)
				npzfile = np.load(path_to_store+'Audio/'+filename+'.npz')
				print('*****AUDIO*****')
				print('feats:', npzfile['feats'].shape) # (n_feats x n_samples)
				print('baf:', npzfile['baf'].shape) 
				print('onset_times:', npzfile['onset_times'].shape) # 
				print()

			count+=1

		except OSError as e:
			print('OSError:', e)
			continue

	# Run global PCA
	Z_total = Z_all[0]
	for i in range(1, len(Z_all)):
		Z_total = np.concatenate((Z_total, Z_all[i]))

	mu = np.average(Z_total, axis=0) # shape 22
	s = np.std(Z_total, axis=0) # shape 22
	print('mu, s:', mu.shape , s.shape)

	Z_total = StandardScaler().fit_transform(Z_total)
	print('Z_total', Z_total.shape) # 22x22
	k = Z_total.shape[1]
	pca = PCA(n_components=10)
	pca.fit(Z_total)
	W = pca.components_ # eigenvectors
	print('W', W.shape) 
	np.savez(path_to_store+'PCA_W_10.npz', W=W, mu=mu, s=s)

	print()


if __name__ == '__main__':
	# Set up command-line argument parsing
	parser = argparse.ArgumentParser(
		description='python data_prep_av_bulk.py',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--fs', default=FS, type=int, action='store',
						help='Global audio sampling rate to use')
	parser.add_argument('--hop', default=HOP, type=int, action='store',
						help='hop length')
	parser.add_argument('--w_size', default=W_SIZE, type=int, action='store',
						help='window size')

	parser.add_argument('--pathToData', type=str, default='../', help='Path to the openpose direcotry')

	parser.add_argument('--method', type=str, default='melspec', help='')
	parser.add_argument('--poly', default=False, type=str2bool, help='Run for polyphonic (True) or monophonic (False) videos')
	parser.add_argument('--strings', type=str2bool, default=True, help='Keep only strings (True) or or keep all instruments (False) videos')
	parser.add_argument('--no_strings', type=str2bool, default=False, help='Keep only strings (True) or or keep all instruments (False) videos')
	parser.add_argument('--visualize', type=str2bool, default= False, help='Visualize skeleton moves.')
	parser.add_argument('--spectralflux', type=str2bool, default= False, help='Visualize skeleton moves.')
	parser.add_argument('--log', type=str2bool, default= False)
	parser.add_argument('--mov_avg', type=str2bool, default= True, help='.')
	parser.add_argument('--awgn', type=str2bool, default= False, help='.')
	parser.add_argument('--HandROIs', type=str2bool, default= True, help='.')
	parser.add_argument('-optflow', action='store_true') # i.e. if -rm is used as argument then args.rmdirs==

	# parser.add_argument('--store', type=str2bool, default= True, help='.')

	# parameters = vars(parser.parse_args(sys.argv[1:]))
	args = parser.parse_args()


	# prepare_data(parameters)
	prepare_data(args)

