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
# from skimage import util
import skimage
from shutil import copyfile
sys.path.append("../")
import copy 
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from utils import str2bool
import utils
# np.set_printoptions(threshold=sys.maxsize)
import math 



FS = 48000
# HOP = 512 
HOP = 480 # 10 ms
W_SIZE = 2048 # 42.7 ms


def prepare_data(args):

	# Create path_ot_store
	path_to_store = args.pathToStore+'/'#+args.audio_feats
	# if args.log:
	# 	path_to_store += 'log'
	# if args.spectralflux:
	# 	path_to_store += 'spectralflux'
	# if args.mov_avg:
	# 	path_to_store += 'mov_avg'
	# path_to_store += '/'

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
		# os.mkdir(path_to_store+'Li')

	# linux
	# path_to_data = "/media/gbastas/New Volume/users/grigoris/Datasets/OpenPoseExtractedData/OpenPoseData_all_hand/"
	path_to_data = args.pathToSkeletons


	# path_to_copy_data_AuSep = '../wav/'
	# path_to_copy_data_Notes = '../annotations/'
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
			keypoints, leftHandKeyPoints, rightHandKeyPoints = utils.scan_folder_sep(path_to_data + subdir)
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
			visual_feats = utils.eliminate_abrupt_keypoint_shifts(visual_feats)
			visual_feats = utils.eliminate_low_confidence_keypoints(conf, visual_feats) # discard points with confidence value  <0.2
			visual_feats = utils.interpolate_keypoints(visual_feats)

			hand_feats = utils.eliminate_abrupt_keypoint_shifts(hand_feats)
			hand_feats = utils.eliminate_low_confidence_keypoints(hand_conf, hand_feats)

			hand_feats_mu = utils.hand_center_mass(hand_feats)
			hand_feats_mu = utils.interpolate_keypoints(hand_feats_mu)

			hand_feats = utils.interpolate_keypoints(hand_feats)

			##### Moving Average ####
			if args.mov_avg:
				visual_feats = utils.centered_moving_average(visual_feats, n=5)

				hand_feats_mu = np.array([np.mean(hand_feats_mu, axis=1) for i in range(hand_feats_mu.shape[1])] ).T # (seq_len x 2)

				print("hand_feats_mu", hand_feats_mu.shape, hand_feats.shape)

				hand_feats = utils.centered_moving_average(hand_feats, n=3)

			Z = utils.get_derivatives(visual_feats).T # speed martix
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
			# urmp_path = img_as_u
			urmp_path = args.pathToURMP
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
					if args.audio_feats=='chromas':
						audio_feats = librosa.feature.chroma_stft(audio_data_mix, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
					elif args.audio_feats=='chromas_norm':
						audio_feats = librosa.feature.chroma_stft(audio_data_mix, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
						audio_feats = librosa.util.normalize(audio_feats, norm=2., axis=1)
					elif args.audio_feats=='melspec':
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
						hand_rois = utils.get_roi(path_to_mkv_file, hand_feats_mu, px=px) # (duration, 100, 100, 3)		
						if px==200:
							hand_rois = [skimage.util.img_as_ubyte(resize(img, (100,100,3))) for img in hand_rois] # rescaled and uint8 
						hand_rois = np.array(hand_rois)

						if args.optflow:
							print('**** OPTICAL FLOW ****')
							hand_rois = utils.get_opt_flow(hand_rois)

			n_audio_frames = audio_feats.shape[1]

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
				baf = np.array( [np.zeros( n_audio_frames )])
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

	parser.add_argument('--pathToStore', type=str, default='../PrepdData/extracted_features/', help='')
	parser.add_argument('--pathToSkeletons', type=str, default='../OpenPoseData_all_hand/', help='Path to the openpose direcotry')
	parser.add_argument('--pathToURMP', type=str, default='../', help='Path to the openpose direcotry')

	parser.add_argument('--audio_feats', type=str, default='melspec', help='')
	# parser.add_argument('--poly', default=False, type=str2bool, help='Run for polyphonic (True) or monophonic (False) videos')
	parser.add_argument('--strings', type=str2bool, default=True, help='Keep only strings (True) or or keep all instruments (False) videos')
	parser.add_argument('--no_strings', type=str2bool, default=False, help='Keep only strings (True) or or keep all instruments (False) videos')
	parser.add_argument('--visualize', type=str2bool, default= False, help='Visualize skeleton moves.')
	parser.add_argument('--spectralflux', type=str2bool, default= False, help='')
	parser.add_argument('--log', type=str2bool, default=False)
	parser.add_argument('--mov_avg', type=str2bool, default= True, help='.')
	parser.add_argument('--awgn', type=str2bool, default= False, help='.')
	parser.add_argument('-HandROIs', action='store_true', help='To extract also features from ROIs')
	parser.add_argument('-optflow', action='store_true', help="Don't just extract standard pixel vaues, instead compute optical flow features") 

	args = parser.parse_args()

	prepare_data(args)

