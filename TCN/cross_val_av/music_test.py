'''
0. Before running the current sricpt one must already have:
		0.1 extracted skeleton poses from the video recordings (check README.md about the proper use of run_multiple_openpose.py).
		0.2 post-processed the skeleton data and must have saved the features data to .npz files (check README.md about the proper use of data_prep_av_bulk.py).
1. employing myutils.py: 
		Get n_folds (e.g. 8) non-overlapping folds containing input and target values for both the visual and the audio features. Post-process features (i.e. deltas & normalization).
		Use data_generator() to load all data separated in folds. 
		Use fold_generator() to yield features, ground truth and filenames separated in terms of train, validation and test set, 
		in order to loop over n_fold models at each epoch and backpropagateeach distinct loss.
2. employing model.py:
		Load and initialize 8 models of the same architecure (i.e. TCN) for each category (visual, audio and fusion).
3. employing run_epoch(), run_final_test(), fold_generator (form myutils.py) and auxil.py:   
		Train, save and test visual & audio models. We do this n_fold times. Then, train, save and test fusion models. We do this n_fold times.
'''

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
import librosa

from model import TCN, TCN_fusion, TCN_Vis_fusion, CNN_TCN, TCN_Pix_Skltn_fusion
from myutils import data_generator, fold_generator
import fixedgutils
from auxil import train, evaluate, str2bool, plot_learning_curve

import numpy as np
import matplotlib.pyplot as plt
from madmom.evaluation import onsets
from madmom.features.onsets import peak_picking, OnsetPeakPickingProcessor
import os 
import random
import warnings
# import adabound
import cv2
import csv

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='AV Onset Detection')
parser.add_argument('--batch_size', type=int, default=10, metavar='N')	
parser.add_argument('--cuda', action='store_false',
					help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
					help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
					help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=200,
					help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
					help='kernel size (old default: 5)')  # TODO: maybe need to discriminate audio and video ksize
parser.add_argument('--levels_aud', type=int, default=4,
					help='# of levels (default: 4)')
parser.add_argument('--levels_vis', type=int, default=9,
					help='# of levels (default: 4)')					
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
					help='report interval')
parser.add_argument('--lr', type=float, default=1e-3,
					help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
					help='optimizer to use Adam')
parser.add_argument('--nhid', type=int, default=256,
					help='number of hidden units per layer (old default: 150)')
parser.add_argument('--data', type=str, default='Nott',
					help='the dataset to run (default: Nott)')
parser.add_argument('--seed', type=int, default=1111)

parser.add_argument('--fs', default=48000, type=int, action='store',
					help='Global audio sampling rate to use')
parser.add_argument('--hop', default=480, type=int, action='store',
					help='hop length')
parser.add_argument('--w_size', default=2048, type=int, action='store',
					help='window size')
parser.add_argument('--input_type', type=str, default='Audio',
					help='can be "Audio", "Visual" or "AudioVisual" features')
parser.add_argument('--scaling', type=str, default='fixed-midhip',
					help='standard or min-max or fixed-midhip')
parser.add_argument('--dilations', type=str2bool, default=True,
					help='Choose weather to use simple cnn or with dilations')
parser.add_argument('--cnn_step', type=int, default=100000, help='how many frames to load together.')


parser.add_argument('--project_dir', type=str, default='../../', help='Path to the project direcotry')
parser.add_argument('--strings', type=str2bool, default=False, help='Keep only strings (True) or or keep all instruments (False) videos')
parser.add_argument('--visualize', type=str2bool, default= False, help='Visualize skeleton moves.')
parser.add_argument('--poly', type=str2bool, help='Run for polyphonic (True) or monophonic (False) videos')
parser.add_argument('--feats_dir', type=str, default='extracted_features')# , help="'melspec', 'chromas', 'chromas_norm', 'melspecSpectralFlux'")
parser.add_argument('--awgn', type=str2bool, default= False, help='.')
parser.add_argument('--onset_window', type=float, default= 0.05, help='50 ms')
parser.add_argument('--deriv', type=str2bool, default=False)
parser.add_argument('--pca', type=int, default=0, help='num of dimensions')

parser.add_argument('-train', action='store_true', help='')
parser.add_argument('--n_folds', default=8, type=int)
parser.add_argument('--cross_val', type=str2bool, default=True)
parser.add_argument('--model_dir', type=str, default='../../models/')
parser.add_argument('--monofold', type=str2bool, default=True)
parser.add_argument('--store_model', type=str2bool, default=True)

parser.add_argument('-train_fusion', action='store_true', help='')

parser.add_argument('--freeze', type=str2bool, default=False)
parser.add_argument('--fusion_epochs', type=int, default=10)
parser.add_argument('--fusion_strat', type=str, default='tcn', help='tcn or fc')
parser.add_argument('--modality', type=str, default='HandROIs', help="'Hand','HandROIs','Body-Hand','Audio', 'Visual', 'AudioVisual'")
parser.add_argument('-multiLoss', action='store_true')

parser.add_argument('--instr', type=str, default='strings')
parser.add_argument('-rescaled', action='store_true', help='rescaled input to match audio feats i.e. 100 fps')
parser.add_argument('-augment', action='store_true', help='')
parser.add_argument('-cbp', action='store_true', help='CompactBilinearPooling')
parser.add_argument('-multiTest', action='store_true', help='Test both tcnA and tcnB')
parser.add_argument('-vnva', action='store_true', help='Only violin and viola')
parser.add_argument('-vn', action='store_true', help='Only violin')
parser.add_argument('-va', action='store_true', help='Only viola')
parser.add_argument('-vc', action='store_true', help='Only violoncelo')
parser.add_argument('-db', action='store_true', help='Only contrabasso')

args = parser.parse_args()

Nf=args.n_folds
if args.monofold:
	Nf=0

def run_epoch(epoch, XFolds, YFolds, TFolds, FFolds, input_type, best_F_measure, extras):
	global imgpath

	for fold_id, (X_train, X_valid, X_test, Y_train, Y_valid, Y_test, T_train, T_valid, T_test, F_train, F_val, F_test) in enumerate( fold_generator(args, XFolds, YFolds, TFolds, FFolds) ):

		if fold_id>Nf: continue #NOTE:
		fig, axs = plt.subplots(len(F_test), 1, figsize=(20,10))

		model_name = "TCN_"+input_type+'_'+str(fold_id)+".pt"
		print("************ Train_"+input_type+'_'+str(fold_id)+" ************")
		eval_objects = train(ep, X_train, Y_train, T_train, args, extras[fold_id], axs, FNames=F_train)
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Training_"+input_type+'_'+str(fold_id)+":\t Precision", precision, "\tRecall", recall, "F_measure", F_measure)

		vloss, eval_objects = evaluate(ep, X_valid, Y_valid, T_valid, args, extras[fold_id]['model'], input_type, fold_id, axs,  name='Validation', FNames=F_val)
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Validation_"+input_type+'_'+str(fold_id)+":\t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)

		# Save best model so far
		if F_measure > best_F_measure[fold_id] and args.store_model:
			model = extras[fold_id]['model']
			with open(args.model_dir+model_name, "wb") as f:
				print('Model saved!', model_name)
				torch.save(model, f)
			best_F_measure[fold_id] = F_measure

		print("************ Test_"+input_type+'_'+str(fold_id)+" ************")
		tloss, eval_objects = evaluate(ep, X_test, Y_test, T_test, args, extras[fold_id]['model'], input_type, fold_id, axs, name='Test', FNames=F_test)
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Test_"+input_type+'_'+str(fold_id)+":\t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)
		print()

		fig.suptitle("TCN_"+input_type+'_'+str(fold_id)+": Precision"+str(precision)+"\tRecall"+str(recall)+str("\tF_measure")+str(F_measure), fontsize=16)	
		fig.savefig(imgpath+'img_'+input_type+'_'+str(ep)+'_'+str(fold_id)+'.png')
		plt.close(fig)

def run_fusion_epoch(epoch, XAFolds, YAFolds, TAFolds, XBFolds, FFolds, input_type, best_F_measure, extras):
	global imgpath
	for fold_id, (AFolds, BFolds) in enumerate( zip(fold_generator(args, XAFolds, YAFolds, TAFolds, FFolds), fold_generator(args, XBFolds, YAFolds, TAFolds, FFolds)) ):	
		if fold_id>Nf: continue # NOTE:

		X_A_train, X_A_valid, X_A_test, Y_train, Y_valid, Y_test, T_train, T_valid, T_test, F_train, F_val, F_test = AFolds
		X_B_train, X_B_valid, X_B_test, _, _, _, _, _, _, _, _, _ = BFolds	
		fig, axs = plt.subplots(8, 1, figsize=(20,10)) # TODO: do it properly

		model_name = args.model_dir+"TCN_"+args.modality+"_fusion_"+str(fold_id)+".pt"

		# print ('Epoch: '+str(ep))
		print("************ Train_Vis_Fusion_"+str(fold_id)+" ************")
		eval_objects = fusion_train(ep, X_A_train, X_B_train, Y_train, T_train, F_train, args, extras[fold_id])
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Training: Precision", precision, "Recall", recall, "F_measure", F_measure)

		vloss, eval_objects = fusion_evaluate(ep, X_A_valid, X_B_valid, Y_valid, T_valid, args, extras[fold_id]['model'], fold_id, axs=axs, name='Validation', FNames=F_val)
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Validation: Precision", precision, "Recall", recall, "F_measure", F_measure)

		# Save best model so far
		if F_measure >= best_F_measure[fold_id] and args.store_model:
			model = extras[fold_id]['model']
			with open(model_name, "wb") as f:
				torch.save(model, f)
			best_F_measure[fold_id] = F_measure

		print("************ Test_Vis_Fusion_"+str(fold_id)+" ************")
		tloss, eval_objects = fusion_evaluate(ep, X_A_test, X_B_test, Y_test, T_test, args, extras[fold_id]['model'], fold_id, axs=axs, name='Test', FNames=F_test)
		evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
		precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
		print("Test:\t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)
		print()

		fig.suptitle("TCN_Vis_Fusion_"+str(fold_id)+": Precision"+str(precision)+"\tRecall"+str(recall)+str("\tF_measure")+str(F_measure), fontsize=16)	
		fig.savefig(imgpath+'img_'+args.modality+'_'+str(ep)+'_'+str(fold_id)+'.png')
		plt.close(fig)			


def run_fusion_final_test(args, XAFolds, YAFolds, TAFolds, XBFolds, FFolds, extras=None):
	global lr
	global imgpath
	print('**************** Run Final Test *****************')

	p_m, r_m, F_m = 0, 0, 0
	with open('../../results/out'+args.modality+'_'+args.fusion_strat+'.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ')
		writer.writerow(["Precision", "Recall", "F_measure"])

		for fold_id, (AFolds, BFolds) in enumerate( zip(fold_generator(args, XAFolds, YAFolds, TAFolds, FFolds), fold_generator(args, XBFolds, YAFolds, TAFolds, FFolds)) ):	
			if fold_id>Nf: continue # NOTE:		

			_, _, X_A_test, _, _, Y_test, _, _, T_test, _, _, F_test = AFolds
			_, _, X_B_test, _, _, _, _, _, _, _, _, _, = BFolds


			print('**************** Run Fusion Final Test *****************')
			fig, axs = plt.subplots(len(F_test), 1, figsize=(20,10))

			if args.fusion_strat == "activations":
				model_name_a = args.model_dir+"TCN_Visual_"+str(fold_id)+".pt"
				print(model_name_a)
				print('-' * 89)
				model_vis = torch.load(open(model_name_a, "rb"))			
				# model_vis = torch.load(open(model_name_a, "rb"))			

				if args.modality=='Body-Hand':
					# model_name_b = args.model_dir+"TCN_Hand_"+str(fold_id)+".pt"
					model_name_b = args.model_dir+"TCN_HandROIs_"+str(fold_id)+".pt"
				elif args.modality=='AudioVisual':
					model_name_b = args.model_dir+"TCN_Audio_"+str(fold_id)+".pt"

				print(model_name_b)
				print('-' * 89)
				model_b = torch.load(open(model_name_b, "rb"))	

				print("************ Test_Vis_Fusion_"+str(fold_id)+" ************")
				tloss, eval_objects = fusion_evaluate(0, X_A_test, X_B_test, Y_test, T_test, args, model_vis, fold_id, axs=axs, name='Test', FNames=F_test, model_b=model_b)
				evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
				precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
				if not args.multiTest:
					print("Test:\t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)
					print()

			if args.fusion_strat in ['fc', 'tcn']:			

				model_name = args.model_dir+"TCN_"+args.modality+"_fusion_"+str(fold_id)+".pt"
		
				print(model_name)
				print('-' * 89)
				model = torch.load(open(model_name, "rb"))

				if args.multiTest:
					model.multiTest = True
					model.multiLoss = False

				print("************ Test Fusion_"+str(fold_id)+" ************")
				# tloss, eval_objects = fusion_evaluate(0, X_A_test, X_B_test, Y_test, T_test, args, extras[fold_id]['model'], fold_id, axs=axs, name='Test', FNames=F_test)
				tloss, eval_objects = fusion_evaluate(0, X_A_test, X_B_test, Y_test, T_test, args, model, fold_id, axs=axs, name='Test', FNames=F_test) 
				evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
				precision, recall, F_measure = evalu[13], evalu[15], float(evalu[17])
				if not args.multiTest:
					print("Test:\t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)
					print()

				fig.suptitle("TCN_Vis_Fusion_"+str(fold_id)+": Precision"+str(precision)+"\tRecall"+str(recall)+str("\tF_measure")+str(F_measure), fontsize=16)	
				fig.savefig(imgpath+'img_'+args.modality+'_final_'+str(fold_id)+'.png')
				plt.close(fig)	

			writer.writerow([str(precision), str(recall), str(F_measure)])

			p_m += float(precision)
			r_m += float(recall)
			F_m += float(F_measure)
				
		print("Average: Precision "+str(round(p_m/args.n_folds,3))+"\tRecall "+str(round(r_m/args.n_folds, 3))+str("\tF_measure ")+str(round(F_m/args.n_folds, 3)))	

		writer.writerow(["Average"])
		writer.writerow([str(round(p_m/args.n_folds,3)), str(round(r_m/args.n_folds,3)), str(round(F_m/args.n_folds,3))])


def run_final_test(args, XFolds, YFolds, TFolds, FFolds, optimizer, input_type):
	global lr
	global imgpath
	print('**************** Run Final Test *****************')

	p_m, r_m, F_m = 0, 0, 0
	with open('../../results/out'+input_type+'.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ')
		writer.writerow(["Precision", "Recall", "F_measure"])

		for fold_id, (_, _, X_test, _, _, Y_test, _, _, T_test, F_train, F_val, F_test) in enumerate( fold_generator(args, XFolds, YFolds, TFolds, FFolds) ):	
			if fold_id>Nf: continue #NOTE:
			print('**************** Run Final Test *****************')
			fig, axs = plt.subplots(len(F_test), 1, figsize=(20,10))		
			model_name = args.model_dir+"TCN_"+input_type+'_'+str(fold_id)+".pt"

			print(model_name)

			print('-' * 89)
			model = torch.load(open(model_name, "rb"))

			tloss, eval_objects = evaluate(None, X_test, Y_test, T_test, args, model, input_type, fold_id, axs, name='FinalTest_'+str(fold_id), FNames=F_test, writer=writer)

			evalu = str(onsets.OnsetMeanEvaluation(eval_objects)).split()
			precision, recall, F_measure = evalu[13], evalu[15], evalu[17]
			print("Final "+input_type+" Test - Fold:"+str(fold_id)+" \t Precision", precision, "\tRecall", recall, "\tF_measure", F_measure)

			fig.suptitle("TCN: Precision"+str(precision)+"\tRecall"+str(recall)+str("\tF_measure")+str(F_measure), fontsize=16)	
			plt.close(fig)

			writer.writerow([str(precision), str(recall), str(F_measure)])

			fig.suptitle("TCN_Vis_Fusion_"+str(fold_id)+": Precision"+str(precision)+"\tRecall"+str(recall)+str("\tF_measure")+str(F_measure), fontsize=16)	
			fig.savefig(imgpath+'img_'+args.modality+'_final_'+str(fold_id)+'.png')
			plt.close(fig)	

			p_m += float(precision)
			r_m += float(recall)
			F_m += float(F_measure)

		print("Average: Precision"+str(p_m/args.n_folds)+"\tRecall"+str(r_m/args.n_folds)+str("\tF_measure")+str(F_m/args.n_folds))	

		writer.writerow(["Average"])
		writer.writerow([str(round(p_m/args.n_folds,3)), str(round(r_m/args.n_folds,3)), str(round(F_m/args.n_folds,3))])


if __name__ == "__main__":

	# if args.model_dir:
	# 	os.mkdir(args.model_dir)

	imgpath = args.project_dir+ 'imgs/'
	args.project_dir += 'PrepdData/' + args.feats_dir + '/' #NOTE important to handle e.g. DATAchormas etc.
	
	# Set the random seed manually for reproducibility.
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	print(args)
	
	# Get n_folds (e.g. 8) non-overlapping folds containing input and target values for both the visual and the audio features
	if args.modality in ['Audio', 'AudioVisual', 'Visual']:
		if args.instr == 'guitar':
			XAudFolds, YAudFolds, TAudFolds, FFolds = fixedgutils.data_generator(args, "Audio")
			from fixedgutils import fold_generator
		else:
			XAudFolds, YAudFolds, TAudFolds, FFolds = data_generator(args, "Audio")
		n_audio_frames = XAudFolds[0][0].shape[0] 


	if args.modality in ['Visual', 'AudioVisual', 'Body-Hand']:
		XVisFolds, YVisFolds, TVisFolds, FFolds = data_generator(args, "Visual")	# (fold_id, track_id, time_bin, feature_iid)

	if args.modality == "Hand":
		XHandFolds, _, _, _ = data_generator(args, "Hand")

	if args.modality == "HandROIs" or args.modality == "Body-Hand":
		XPixFolds, YPixFolds, TPixFolds, FPixFolds = data_generator(args, "HandROIs") # foldID x sample x torch.Size([n_frames, 100, 100, 3])


	# NOTE:
	# print('XHandShape', len(XHandFolds), len(XHandFolds[0]), XHandFolds[0][0].shape)
	# for i in range(len(XHandFolds)):
	# 	for j in range(len(XHandFolds[i])):
	# 		XVisFolds[i][j] = torch.cat((XVisFolds[i][j].T, XHandFolds[i][j].T)).T

	# NOTE:
	if args.modality == "Hand":
		XVisFolds = XHandFolds


	print()
	# Feat Size
	if args.modality in ['Visual', 'AudioVisual', 'Body-Hand']: visual_input_size = XVisFolds[0][0].shape[1] # e.g 66a
	if args.modality == "Hand": hand_input_size = XHandFolds[0][0].shape[1] # e.g. 126
	if args.modality in ['Audio', 'AudioVisual']: audio_input_size = XAudFolds[0][0].shape[1] # e.g. 40

	print()
	if args.modality in ['Visual', 'AudioVisual']: print("visual_input_size", visual_input_size)
	if args.modality == "Hand": print("hand_input_size", hand_input_size)
	if args.modality in ['Audio', 'AudioVisual']: print("audio_input_size", audio_input_size)
	print()

	# Some hyperparameters
	output_size = 2 
	if args.modality in ['Visual', 'AudioVisual', 'Hand', 'HandROIs', 'Body-Hand']:
		n_visual_channels = [args.nhid] * args.levels_vis # e.g. [150] * 9
		n_hand_channels = n_visual_channels
	if args.modality in ['Audio', 'AudioVisual']: n_audio_channels = [args.nhid] * args.levels_aud # e.g. [150] * 4
	kernel_size = args.ksize 
	dropout = args.dropout

	# Initialize n_fold model instances for each category: visual, audio, fusion
	extras_pixel = []
	extras_visual=[]
	extras_audio=[]
	lr = args.lr
	for i in range(args.n_folds):
		if args.modality in ['Visual', 'AudioVisual']:
			visual_model = TCN(visual_input_size, output_size, n_visual_channels, kernel_size, dropout=args.dropout, dilations=args.dilations)
		if args.modality in ["HandROIs", "Body-Hand"]:
			pixel_model = CNN_TCN(args, output_size, n_visual_channels, kernel_size, dropout)
		#NOTE:
		if args.modality == 'Body-Hand':
			visual_model_name = args.model_dir+'TCN_Body_'+str(i)+'.pt'
			hand_model_name = args.model_dir+'TCN_Hand_'+str(i)+'.pt'	
			pixel_model_name = args.model_dir+'TCN_HandROIs_'+str(i)+'.pt'
			# visual_model = TCN_Vis_fusion(args, visual_input_size, hand_input_size, visual_model_name, hand_model_name, output_size, n_visual_channels, n_hand_channels, kernel_size, dropout=args.dropout, dilations=args.dilations)
			pixel_model = TCN_Pix_Skltn_fusion(args, visual_input_size, visual_model_name, pixel_model_name, output_size, n_visual_channels, n_hand_channels, kernel_size, dropout=args.dropout, dilations=args.dilations)
		if args.modality in ['Audio', 'AudioVisual']:
			audio_model  = TCN(audio_input_size, output_size, n_audio_channels, kernel_size, dropout=args.dropout, dilations=args.dilations)

		if args.cuda:
			if args.modality in ["HandROIs", "Body-Hand"]: pixel_model.cuda()
			if args.modality in ['Visual', 'AudioVisual']: visual_model.cuda()
			if args.modality in ['Audio', 'AudioVisual']: audio_model.cuda()

		if args.modality in ["HandROIs", "Body-Hand"]: pixel_optimizer = getattr(optim, args.optim)(pixel_model.parameters(), lr=lr)#, weight_decay=1e-5)
		if args.modality in ['Visual', 'AudioVisual']: visual_optimizer = getattr(optim, args.optim)(visual_model.parameters(), lr=lr)
		if args.modality in ['Audio', 'AudioVisual']: audio_optimizer = getattr(optim, args.optim)(audio_model.parameters(), lr=lr)


		if args.modality in ["HandROIs", "Body-Hand"]: extras_pixel += [{'model':pixel_model, 'lr':lr, 'optimizer':pixel_optimizer}]
		if args.modality in ['Visual', 'AudioVisual']: extras_visual += [{'model':visual_model, 'lr':lr, 'optimizer':visual_optimizer}]
		if args.modality in ['Audio', 'AudioVisual']: extras_audio += [{'model':audio_model, 'lr':lr, 'optimizer':audio_optimizer}]

	if args.modality in ["HandROIs", "Body-Hand"]: best_pixel_F_measure = [-0.1] * args.n_folds
	best_visual_F_measure = [-0.1] * args.n_folds
	best_audio_F_measure = [-0.1] * args.n_folds

	# Train Visual and Audio Models
	if args.train:
		for ep in range(1, args.epochs+1):
			print ('Epoch: '+str(ep))
			################################## VISUAL #######################################
			if args.modality in ['Visual', 'AudioVisual']: # TODO: fixed if args.rescaled for baf
				if args.rescaled: YVisFolds = YAudFolds
				run_epoch(ep, XVisFolds, YVisFolds, TVisFolds, FFolds, 'Visual', best_visual_F_measure, extras_visual)

			if args.modality == 'Hand':
				run_epoch(ep, XVisFolds, YVisFolds, TVisFolds, FFolds, 'Hand', best_visual_F_measure, extras_visual)

			if args.modality == "HandROIs":
				run_epoch(ep, XPixFolds, YPixFolds, TPixFolds, FPixFolds, 'HandROIs', best_pixel_F_measure, extras_pixel)

			if args.modality == 'Body-Hand':
				run_fusion_epoch(ep, XVisFolds, YPixFolds, TPixFolds, XPixFolds, FFolds, 'Body-Hand', best_visual_F_measure, extras_pixel)

			################################## AUDIO #######################################
			if args.modality in ['Audio', 'AudioVisual']:
				run_epoch(ep, XAudFolds, YAudFolds, TAudFolds, FFolds, 'Audio', best_audio_F_measure, extras_audio)
				
	################################## VISUAL Final Test #######################################
	if args.modality in ['Visual', 'AudioVisual']:
		if args.rescaled: YVisFolds = YAudFolds
		run_final_test(args, XVisFolds, YVisFolds, TVisFolds, FFolds, visual_optimizer, 'Visual') 
	if args.modality == 'Hand':
		run_final_test(args, XVisFolds, YVisFolds, TVisFolds, FFolds, visual_optimizer, 'Hand')
	if args.modality == 'Body-Hand':
		run_fusion_final_test(args, XVisFolds, YVisFolds, TVisFolds, XHandFolds, FFolds) 
	if args.modality == "HandROIs":
		run_final_test(args, XPixFolds, YPixFolds, TPixFolds, FPixFolds, pixel_optimizer, 'HandROIs')		
	################################## AUDIO Final Test  #######################################
	if args.modality in ['Audio', 'AudioVisual']:
		run_final_test(args, XAudFolds, YAudFolds, TAudFolds, FFolds, audio_optimizer, 'Audio')			

