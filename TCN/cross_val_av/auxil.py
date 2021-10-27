from torch.autograd import Variable
import torch.nn as nn
import torch
from madmom.features.onsets import peak_picking, OnsetPeakPickingProcessor
import librosa
from madmom.evaluation import onsets
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import sys
np.set_printoptions(threshold=sys.maxsize)


def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0.7, 1)):
	plt.figure()
	plt.title("Learning Curve")
	if ylim is not None:
		plt.ylim(*ylim)
	# plt.ylim(0,1)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	plt.plot(train_sizes, train_scores, '-', color="r",
			label="Training score")
	plt.plot(train_sizes, test_scores, '-', color="g",
			label="Validation score")

	plt.legend(loc="best")

	return plt


def evaluate(ep, X_data, Y_data, T_data, args, model, input_type, fold_id, axs, name='Eval', FNames=[], writer=None):

	# model, _, _ = extras['model'], extras['lr'], extras['optimizer']
	model.eval()
	eval_idx_list = np.arange(len(X_data), dtype="int32")
	total_loss = 0.0
	count = 0
	EvalObjects = []
	with torch.no_grad():

		for idx in eval_idx_list:
			
			if args.vnva and FNames[idx].split('_')[1] not in ['vn','va']:
				continue

			if args.vn and FNames[idx].split('_')[1] not in ['vn']: 
				continue

			if args.va and FNames[idx].split('_')[1] not in ['va']: 
				continue

			if args.vc and FNames[idx].split('_')[1] not in ['vc']: 
				continue

			if args.db and FNames[idx].split('_')[1] not in ['db']: 
				continue

			x, y = Variable(X_data[idx], requires_grad=True), Variable(Y_data[idx], requires_grad=False) # _greg_
			if args.cuda:
				# x, y = x.cuda(), y.cuda()
				y = y.cuda()

			##########################################################
			if args.modality!='HandROIs':
				if args.cuda: x = x.cuda()
				output = model(x.unsqueeze(0))
			else:
				output = model(x)
			
			lossObj = nn.BCELoss() # _greg_

			if args.modality == 'Visual':
				y = torch.cat([y, y[-1].unsqueeze(0)], dim=0) 
			loss = lossObj(output, y.unsqueeze(0).double())

			total_loss += loss.item()
			count += output.size(0)

			# EVALUATE
			o = output.squeeze(0).cpu().detach()
			y = y.cpu().detach()

			if not args.rescaled:
				oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
				otimes = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
			else:
				oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
				otimes = librosa.core.frames_to_time(oframes, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
				# otimes = librosa.core.frames_to_time(oframes, sr=args.fs, hop_length=args.hop)
			
			annotations=T_data[idx]
			EvalObjects.append( onsets.OnsetEvaluation(otimes, annotations, window=args.onset_window) )

			# VISUALIZE
			if 'Test' in name:# and idx<10:  # _greg_

				axs[idx].plot(o[:,0])
				axs[idx].plot(y[:,0], alpha=0.5)
				axs[idx].set_ylabel(FNames[idx], fontsize=5)

		eval_loss = total_loss #/ count
		return eval_loss, EvalObjects


def train(ep, X_train, Y_train, T_train, args, extras, axs, FNames=[]):
	batch_size = args.batch_size
	model, lr, optimizer = extras['model'], extras['lr'], extras['optimizer']
	model.train()
	total_loss = 0
	count = 0
	# step = 4000
	tp=0
	EvalObjects=[]
	for idx in range(len(X_train)):

		if args.vnva and FNames[idx].split('_')[1] not in ['vn','va']:
			continue

		x, y = Variable(X_train[idx], requires_grad=True), Variable(Y_train[idx], requires_grad=False) # _greg_
		if args.cuda: y = y.cuda()

		optimizer.zero_grad()
		lossObj = nn.BCELoss() # _greg_

		if args.modality!='HandROIs':
			if args.cuda: x = x.cuda()
			output = model(x.unsqueeze(0))
		else:
			output = model(x)

		lossObj = nn.BCELoss() # _greg_

		if args.modality == 'Visual':
			y = torch.cat([y, y[-1].unsqueeze(0)], dim=0) 
		loss = lossObj(output, y.unsqueeze(0).double())
		total_loss += loss.item() # NOTE ?
		count += output.size(0)

		if args.clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

		loss.backward()
		optimizer.step()

		# EVALUATE
		o = output.squeeze(0).cpu().detach()
		y = y.cpu().detach()

		if not args.rescaled:
			oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
			otimes = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
		else:
			oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
			otimes = librosa.core.frames_to_time(oframes, sr=args.fs, n_fft=args.w_size, hop_length=args.hop) 
			# otimes = librosa.core.frames_to_time(oframes, sr=args.fs, hop_length=args.hop) 

		ground_truth=T_train[idx]

		EvalObjects.append( onsets.OnsetEvaluation(otimes, ground_truth, window=args.onset_window) )

		output.cpu().detach()

	return EvalObjects


