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

			y = torch.cat([y, y[-1].unsqueeze(0)], dim=0) # NOTE
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

		y = torch.cat([y, y[-1].unsqueeze(0)], dim=0) # NOTE
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


# # onset evaluation function (madmom variant)
# WINDOW=0.05 # __greg__
# def my_onset_evaluation(detections, annotations, window=WINDOW):
# 	"""
# 	Determine the true/false positive/negative detections.

# 	Parameters
# 	----------
# 	detections : numpy array
# 		Detected notes.
# 	annotations : numpy array
# 		Annotated ground truth notes.
# 	window : float, optional
# 		Evaluation window [seconds].

# 	Returns
# 	-------
# 	tp : numpy array, shape (num_tp,)
# 		True positive detections.
# 	fp : numpy array, shape (num_fp,)
# 		False positive detections.
# 	tn : numpy array, shape (0,)
# 		True negative detections (empty, see notes).
# 	fn : numpy array, shape (num_fn,)
# 		False negative detections.
# 	errors : numpy array, shape (num_tp,)
# 		Errors of the true positive detections wrt. the annotations.

# 	Notes
# 	-----
# 	The returned true negative array is empty, because we are not interested
# 	in this class, since it is magnitudes bigger than true positives array.

# 	"""
# 	# make sure the arrays have the correct types and dimensions
# 	detections = np.asarray(detections, dtype=np.float)
# 	annotations = np.asarray(annotations, dtype=np.float)
# 	# TODO: right now, it only works with 1D arrays
# 	if detections.ndim > 1 or annotations.ndim > 1:
# 		raise NotImplementedError('please implement multi-dim support')

# 	# init TP, FP, FN and errors
# 	tp = np.zeros(0)
# 	fp = np.zeros(0)
# 	tn = np.zeros(0)  # we will not alter this array
# 	fn = np.zeros(0)
# 	my = np.zeros(0)
# 	errors = np.zeros(0)

# 	# if neither detections nor annotations are given
# 	if len(detections) == 0 and len(annotations) == 0:
# 		# return the arrays as is
# 		return tp, fp, tn, fn, errors
# 	# if only detections are given
# 	elif len(annotations) == 0:
# 		# all detections are FP
# 		return tp, detections, tn, fn, errors
# 	# if only annotations are given
# 	elif len(detections) == 0:
# 		# all annotations are FN
# 		return tp, fp, tn, annotations, errors

# 	# window must be greater than 0
# 	if float(window) <= 0:
# 		raise ValueError('window must be greater than 0')

# 	# sort the detections and annotations
# 	det = np.sort(detections)
# 	ann = np.sort(annotations)
# 	# cache variables
# 	det_length = len(detections)
# 	ann_length = len(annotations)
# 	det_index = 0
# 	ann_index = 0
# 	# iterate over all detections and annotations
# 	while det_index < det_length and ann_index < ann_length:
# 		# fetch the first detection
# 		d = det[det_index]
# 		# fetch the first annotation
# 		a = ann[ann_index]
# 		# compare them
# 		if abs(d - a) <= window:
# 			# TP detection
# 			tp = np.append(tp, d)
# 			# __greg__
# 			my = np.append(my, 1)
# 			# append the error to the array
# 			errors = np.append(errors, d - a)
# 			# increase the detection and annotation index
# 			det_index += 1
# 			ann_index += 1
# 		elif d < a:
# 			# FP detection
# 			fp = np.append(fp, d)
# 			# increase the detection index
# 			det_index += 1
# 			# do not increase the annotation index
# 		elif d > a:
# 			# we missed a annotation: FN
# 			fn = np.append(fn, a)
# 			# __greg__
# 			my = np.append(my, 0)
# 			# do not increase the detection index
# 			# increase the annotation index
# 			ann_index += 1
# 		else:
# 			# can't match detected with annotated onset
# 			raise AssertionError('can not match % with %', d, a)
# 	# the remaining detections are FP
# 	fp = np.append(fp, det[det_index:])
# 	# the remaining annotations are FN
# 	fn = np.append(fn, ann[ann_index:])
# 	# check calculations
# 	if len(tp) + len(fp) != len(detections):
# 		raise AssertionError('bad TP / FP calculation')
# 	if len(tp) + len(fn) != len(annotations):
# 		raise AssertionError('bad FN calculation')
# 	if len(tp) != len(errors):
# 		raise AssertionError('bad errors calculation')
# 	# convert to numpy arrays and return them
# 	return np.array(my), np.array(tp), np.array(fp), tn, np.array(fn), np.array(errors)
# 	# return np.array(my)


# def fusion_evaluate(ep, X_visual_data, X_audio_data, Y_data, T_data, args, model, fold_id, axs=None, name='Eval', FNames=[], model_b=''):
# 	if model_b: # that is if fusion_strat=="activations"
# 		model_a = model
# 	model.eval()
# 	eval_idx_list = np.arange(len(X_visual_data), dtype="int32")
# 	total_loss = 0.0
# 	count = 0
# 	EvalObjects = []
# 	n=0
# 	total_orRc = 0
# 	total_andRc = 0
# 	with torch.no_grad():
# 		for idx in eval_idx_list:



# 			if args.vnva and FNames[idx].split('_')[1] not in ['vn','va']:
# 				continue


# 			if args.vn and FNames[idx].split('_')[1] not in ['vn']: 
# 				continue

# 			n+=1 # counter

# 			x1, x2, y = Variable(X_visual_data[idx], requires_grad=True), Variable(X_audio_data[idx], requires_grad=True), Variable(Y_data[idx], requires_grad=False) # _greg_
# 			if args.cuda:
# 				x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

# 			if model_b: # that is if fusion_strat=="activations"
# 				# output_body, output_hand = model_a(x1.unsqueeze(0)), model_b(x2.unsqueeze(0))	
# 				output_body, output_hand = model_a(x1.unsqueeze(0)), model_b(x2)	
# 				output = output_body # that's just a trick
# 			else:
# 				if args.multiLoss or args.multiTest:
# 					# output, output_body, output_hand = model(x1.unsqueeze(0), x2.unsqueeze(0))
# 					x2 = torch.cat([x2, x2[-1].unsqueeze(0)], dim=0) # NOTE: for optflow
# 					output, output_body, output_hand = model(x1.unsqueeze(0), x2)
# 				else:
# 					# output = model(x1.unsqueeze(0), x2.unsqueeze(0))
# 					x2 = torch.cat([x2, x2[-1].unsqueeze(0)], dim=0) # NOTE: for optflow
# 					output = model(x1.unsqueeze(0), x2)

# 			lossObj = nn.BCELoss() # _greg_
# 			# y = torch.cat([y, y[-1].unsquzeeze(0)], dim=0) # NOTE
# 			loss = lossObj(output, y.unsqueeze(0).double())				

# 			if args.multiLoss or args.multiTest:
# 				loss_body = lossObj(output_body, y.unsqueeze(0).double())
# 				loss_hand = lossObj(output_hand, y.unsqueeze(0).double())
# 				loss = loss + loss_body + loss_hand

# 			total_loss += loss.item()
# 			count += output.size(0)

# 			# EVALUATE
# 			if args.fusion_strat == 'activations':
# 				o_a = output_body.squeeze(0).cpu().detach()
# 				o_b = output_hand.squeeze(0).cpu().detach()
# 				o_b = torch.cat([o_b, o_b[-1,:].unsqueeze(0)], dim=0) # NOTE: for optflow

# 				###### MAX ######
# 				# # print(o_a.size(), o_b.size())
# 				# cat = torch.stack([o_a[:,0], o_b[:,0]]) 
# 				# # print(cat.size())
# 				# o = torch.max(cat, dim=0)[0]
# 				# # o  = (o_a[:,0] + o_b[:,0])/2
# 				# if not args.rescaled:
# 				# 	# oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 				# 	# oframes = peak_picking(activations=o[0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 				# 	oframes = peak_picking(activations=o.numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 				# 	otimes = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
# 				# else:
# 				# 	# oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 				# 	oframes = peak_picking(activations=o.numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 				# 	otimes = librosa.core.frames_to_time(oframes, sr=args.fs, hop_length=args.hop)

# 				###### COMBINE ########
# 				o_b = (o_a + o_b)/2
# 				if not args.rescaled:
# 					oframes_a = peak_picking(activations=o_a[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 					otimes_a = librosa.core.samples_to_time(oframes_a, sr=29.97) # predicted onest times
# 					oframes_b = peak_picking(activations=o_b[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 					otimes_b = librosa.core.samples_to_time(oframes_b, sr=29.97) # predicted onest times
# 				else:
# 					oframes_a = peak_picking(activations=o_a[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 					# otimes_a = librosa.core.frames_to_time(oframes_a, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
# 					otimes_a = librosa.core.frames_to_time(oframes_a, sr=args.fs, hop_length=args.hop)
# 					oframes_b = peak_picking(activations=o_b[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 					# otimes_b = librosa.core.frames_to_time(oframes_b, sr=args.fs, n_fft=args.w_size, hop_length=args.hop)
# 					otimes_b = librosa.core.frames_to_time(oframes_b, sr=args.fs, hop_length=args.hop)

# 				# Merge body (ta) and hand (tb) onset times (give priority to body)
# 				otimes=[]
# 				i, j, b = 0, 0, 0
# 				while 2+2==4:
# 					ta = otimes_a[i]
# 					tb = otimes_b[j]
# 					if ta<=tb:
# 						otimes.append(ta)
# 						i+=1
# 					elif ta>tb:
# 						j+=1
# 						# if ta-tb > 10.2: # filter out 
# 						if ta-tb > 1: # filter out 
# 							otimes.append(tb)
# 							b+=1
# 					if j>=len(otimes_b) and i<=len(otimes_a):
# 						otimes += list(otimes_a[i:])
# 						break
# 					if i>=len(otimes_a):# and j>=len(otimes_b):
# 						break

# 				print(b, b/i, i , len(otimes_a))

# 			else: # STANDARD OUTPUT EVALUATION
# 				o = output.squeeze(0).cpu().detach()
				
# 				if not args.rescaled:
# 					oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 					otimes = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
# 				else:
# 					oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 					otimes = librosa.core.frames_to_time(oframes, sr=args.fs, hop_length=args.hop)

# 			y = y.cpu().detach()
			
# 			annotations=T_data[idx]
# 			EvalObjects.append( onsets.OnsetEvaluation(otimes, annotations, window=args.onset_window) )

# 			# VISUALIZE
# 			if 'Test' in name: # _greg_
# 				if args.multiTest:
# 					o_a = output_body.squeeze(0).cpu().detach()
# 					o_b = output_hand.squeeze(0).cpu().detach()	

# 					# print(o_a)
# 					# print(o_b)

# 					oframes = peak_picking(activations=o_a[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 					otimes_a = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
					
# 					oframes = peak_picking(activations=o_b[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 					otimes_b = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times

# 					# print(otimes_a)
# 					# print(otimes_b)

# 					A, tpA, fpA, _, fnA, errorsA = my_onset_evaluation(otimes_a, annotations)
# 					B, tpB, fpB, _, fnB, errorsB = my_onset_evaluation(otimes_b, annotations)

# 					evalu = str(onsets.OnsetEvaluation(otimes_a, annotations, window=args.onset_window)).split()
# 					precisionA, recallA, F_measureA = evalu[9], evalu[11], evalu[13]

# 					evalu = str(onsets.OnsetEvaluation(otimes_b, annotations, window=args.onset_window)).split()
# 					precisionB, recallB, F_measureB = evalu[9], evalu[11], evalu[13]
# 					print(FNames[idx])
# 					print("Pr", precisionA, " ", precisionB) 		
# 					print("Rc", recallA, " ", recallB)
# 					print("Fm", F_measureA, " ", F_measureB)
# 					print("tp",len(tpA),'\t',len(tpB))
# 					# print("tp",len(tpA)/len(A),'\t',len(tpB)/len(A))
# 					# print("fn",len(fnA)/len(A),'\t',len(fnB)/len(A))
# 					with open('./insights/'+FNames[idx], 'w') as f:
# 						f.write('Skeleton\tPixel\n')
# 						f.write('\n')
# 						f.write("Pr "+ precisionA+ " "+ precisionB+'\n') 	
# 						f.write("Rc "+ recallA+ " "+ recallB+'\n')
# 						f.write("Fm "+ F_measureA+ " "+ F_measureB+'\n')
# 						f.write('\n')
# 						f.write('tp:\n')
# 						f.write(str(len(tpA))+'\t'+str(len(tpB))+'\n')
# 						f.write('\n')
# 						f.write('fn:\n')
# 						f.write(str(len(fnA))+'\t'+str(len(fnB))+'\n')
# 						f.write('\n')
# 						cor, cand = 0, 0
# 						for i, j in zip(A, B):
# 							logor = int(i or j)
# 							logand= int(i and j)
# 							# print(logor, logand)
# 							# string = str(int(i))+'\t'+str(int(j))+'\n'
# 							string = str(int(i))+'\t'+str(int(j))+'\t'+str(logor)+'\t'+str(logand)+'\n'
# 							if logor:  cor += 1
# 							if logand: cand += 1
# 							f.write(string)

# 						orRc = cor/len(A)
# 						andRc = cand/len(A)
# 						f.write('\n')
# 						f.write('orTp: '+str(cor))
# 						f.write('andTp: '+str(cand))
# 						f.write('orRc: '+str(orRc))
# 						f.write('andRc: '+str(andRc))

# 					print('orTp:',logor)
# 					print('andTp:',logor)
# 					print('orRc:',orRc)
# 					print('andRc:',andRc)
# 					print()

# 					total_andRc += andRc
# 					total_orRc  += orRc

# 					y = y.cpu().detach() 
# 					# print(o_a.size)
# 					# axs[idx].plot(o_a[:4000,0], alpha=0.5)
# 					# axs[idx].plot(o_b[:4000,0], alpha=0.7)
# 					# axs[idx].plot(y[:4000,0], alpha=0.3)
# 					axs[idx].plot(o_a[1000:3000,0], alpha=0.5)
# 					axs[idx].plot(o_b[1000:3000,0], alpha=0.5)
# 					axs[idx].plot(y[1000:3000,0], alpha=0.3)

# 					axs[idx].set_ylabel(FNames[idx], fontsize=6)
# 				else:
# 					o = output.cpu().detach()
# 					y = y.cpu().detach()
# 					axs[idx].plot(o[0,:4000,0])
# 					axs[idx].plot(y[:4000,0], alpha=0.5)
# 					axs[idx].set_ylabel(FNames[idx], fontsize=6)


# 		if args.multiTest:
# 			MorRecall = round(total_orRc/n,3)
# 			MandRecall = round(total_andRc/n,3)
# 			print('Mean orRecall:', MorRecall)
# 			print('Mean andRecall:', MandRecall)
# 			print()
# 			with open('res'+args.modality+'_'+args.fusion_strat+'.csv', 'a', newline='') as csvfile: 
# 				writer = csv.writer(csvfile, delimiter='\t')
# 				# writer.writerow(["Mean orRecall", "Mean andRecall"])
# 				writer.writerow([str(MorRecall), str(MandRecall)])

# 		eval_loss = total_loss / count
# 		# print(name + " loss: {:.5f}".format(eval_loss))
# 		return eval_loss, EvalObjects


# def fusion_train(ep, X_visual_train, X_audio_train, Y_train, T_train, FNames, args, extras):

# 	# print ('epoch='+str(ep))
# 	model, lr, optimizer = extras['model'], extras['lr'], extras['optimizer']
# 	model.train()
# 	total_loss = 0
# 	count = 0
# 	tp=0
# 	EvalObjects=[]
# 	for idx in range(len(X_visual_train)):

# 		# print(FNames)
# 		if args.vnva and FNames[idx].split('_')[1] not in ['vn','va']: 
# 			continue

# 		if args.vn and FNames[idx].split('_')[1] not in ['vn']: 
# 			continue


# 		x1, x2, y = Variable(X_visual_train[idx], requires_grad=True), Variable(X_audio_train[idx], requires_grad=True), Variable(Y_train[idx], requires_grad=False) # _greg_
# 		if args.cuda:
# 			x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

# 		optimizer.zero_grad()
# 		if args.multiLoss:
# 			# output, output_body, output_hand = model(x1.unsqueeze(0), x2.unsqueeze(0))
# 			x2 = torch.cat([x2, x2[-1].unsqueeze(0)], dim=0) # NOTE: for optflow
# 			# print(x1.size(), x2.size())
# 			output, output_body, output_hand = model(x1.unsqueeze(0), x2)
# 		else:
# 			# output = model(x1.unsqueeze(0), x2.unsqueeze(0))
# 			x2 = torch.cat([x2, x2[-1].unsqueeze(0)], dim=0) # NOTE: for optflow
# 			output = model(x1.unsqueeze(0), x2)

# 		lossObj = nn.BCELoss() # _greg_
# 		# print('AAAA', output.size(), y.size())
# 		# y = torch.cat([y, y[-1].unsqueeze(0)], dim=0) # NOTE
# 		loss = lossObj(output, y.unsqueeze(0).double())
# 		if args.multiLoss:# or args.multiTest:
# 			loss_body = lossObj(output_body, y.unsqueeze(0).double())
# 			loss_hand = lossObj(output_hand, y.unsqueeze(0).double())
# 			loss = loss + loss_body + loss_hand

# 		total_loss += loss.item() # NOTE: ?
# 		count += output.size(0)

# 		if args.clip > 0:
# 			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

# 		loss.backward()
# 		optimizer.step()
# 		if idx > 0 and idx % args.log_interval == 0:
# 			cur_loss = total_loss / count
# 			print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
# 			total_loss = 0.0
# 			count = 0

# 		# EVALUATE
# 		o = output.squeeze(0).cpu().detach()
# 		y = y.cpu().detach()

# 		if not args.rescaled:
# 			oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=1, post_max=1) # madmom method
# 			otimes = librosa.core.samples_to_time(oframes, sr=29.97) # predicted onest times
# 		else:
# 			oframes = peak_picking(activations=o[:,0].numpy(), threshold=0.5, pre_max=2, post_max=2) # madmom method
# 			otimes = librosa.core.frames_to_time(oframes, sr=args.fs, hop_length=args.hop)

# 		ground_truth=T_train[idx]

# 		EvalObjects.append( onsets.OnsetEvaluation(otimes, ground_truth, window=args.onset_window) )

# 	return EvalObjects

