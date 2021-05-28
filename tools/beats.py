'''
Created on 16 Dec 2018

@author: aggelos
'''
import numpy
from scipy.stats import norm


def readAnnotationFile(inFile):
	inf = open(inFile)
	
	retval = []
	lines = inf.readlines()
	for line in lines:
		values = line.split(' ')
		beatval = float(values[0])
		retval.append(beatval)
	inf.close()
	
	return retval

def extract_beat_vector(beat_times,win_size,win_step,frames_n,fr,concat=False):
	if concat == False:
		first_time = numpy.float32(win_size)/fr/2.0
	else:
		first_time = 0
	step_time = numpy.float32(win_step)/fr
	times = first_time+step_time*numpy.arange(0,frames_n)
	beat_activation = numpy.zeros((frames_n,1))
	for b in beat_times:
		diff = numpy.abs(b-times)
		pos = numpy.argmin(diff)
		beat_activation[pos]=1.0
		
	return beat_activation


def smooth_baf(baf, smooth_factor):
	baf = numpy.reshape(baf, (baf.shape[0],))
	locations = numpy.nonzero(baf)[0]
	if (len(locations) > 2):
		diff = locations - numpy.roll(locations, 1)
		diff = diff[1:-1]
		medval = numpy.median(diff)
		# print(medval, smooth_factor)

		vals = numpy.arange(-medval / smooth_factor, medval / smooth_factor, 1)

		normpdf = norm.pdf(vals, 0, medval / smooth_factor)

		z = numpy.convolve(baf, normpdf, mode='same')
		z = z / numpy.max(z)
		z = numpy.reshape(z, (z.shape[0], 1))

	else:
		medval = 0.0
		z = numpy.zeros(shape=(baf.shape[0], 1))

	return z, medval


def smooth_baf_adj(baf, target_mean=0.5):

	prev_smooth_factor = 0
	smooth_factor = 10.0
	signs = []

	while(True):

		[x, medval] = smooth_baf(baf, smooth_factor)
		meanval = numpy.mean(x)
		if numpy.abs(meanval - target_mean) < 0.05:
			return x, medval
		if meanval > target_mean:
			smooth_factor = smooth_factor*1.05

		else:
			smooth_factor = smooth_factor/1.05

def square_baf_const(baf, const_factor, v=1):
	const_factor = int(const_factor)
	baf = numpy.reshape(baf, (baf.shape[0],))
	baf_len = baf.shape[0]
	locations = numpy.nonzero(baf)[0]
	locations_n = len(locations)
	out = numpy.zeros((baf.shape[0],1))
	if (locations_n > 2):
		diff = locations - numpy.roll(locations, 1)
		diff = diff[1:-1]
		medval = numpy.median(diff)

		for l in range(locations_n):

			pos = locations[l]
			prev_idx = pos - const_factor
			next_idx = pos + const_factor

			if (prev_idx < 0):
				prev_idx = 0
			if (next_idx > baf_len):
				next_idx = baf_len
			out[prev_idx:next_idx] = v

	else:
		medval = 0.0
		out = numpy.zeros(shape=(baf.shape[0], 1))

	return out, medval


def square_baf(baf, smooth_factor):
	baf = numpy.reshape(baf, (baf.shape[0],))
	baf_len = baf.shape[0]
	locations = numpy.nonzero(baf)[0]
	locations_n = len(locations)
	out = numpy.zeros((baf.shape[0],1))
	if (locations_n > 2):
		diff = locations - numpy.roll(locations, 1)
		diff = diff[1:-1]
		medval = numpy.median(diff)

		for l in range(locations_n):

			if (l == 0):
				diff_pos = locations[l + 1] - locations[l]
				anoxh_pos = int(diff_pos / smooth_factor)
				anoxh_neg = anoxh_pos
			elif (l == locations_n - 1):
				diff_neg = locations[l] - locations[l - 1]
				anoxh_neg = int(diff_neg / smooth_factor)
				anoxh_pos = anoxh_neg
			else:
				diff_pos = locations[l + 1] - locations[l]
				anoxh_pos = int(diff_pos / smooth_factor)
				diff_neg = locations[l] - locations[l - 1]
				anoxh_neg = int(diff_neg / smooth_factor)

			if (anoxh_pos < 3):
				anoxh_pos = 3
			if (anoxh_neg < 3):
				anoxh_neg = 3

			pos = locations[l]
			prev_idx = pos - anoxh_neg
			next_idx = pos + anoxh_pos

			if (prev_idx < 0):
				prev_idx = 0
			if (next_idx > baf_len):
				next_idx = baf_len

			out[prev_idx:next_idx] = 1

	else:
		medval = 0.0
		out = numpy.zeros(shape=(baf.shape[0], 1))

	return out, medval


def getTempoClass(period, min_period, max_period):

	if period < min_period:
		output=min_period
	elif period > max_period:
		output = max_period
	else:
		output = period
	return output - min_period


def get_tempo_curve(baf, min_period=20, max_period=300):

	baf = numpy.reshape(baf, (baf.shape[0],))
	tempo_curve = numpy.zeros((baf.shape[0],1))
	tempo_curve_dig = numpy.zeros((baf.shape[0], max_period-min_period+1))
	locations = numpy.nonzero(baf)[0]
	beats_n = len(locations)
	if (beats_n > 2):
		diff = locations - numpy.roll(locations, 1)
		diff = diff[1::]

		for b in range(1,beats_n):
			end_pos = locations[b]
			start_pos = locations[b-1]
			period = diff[b-1]
			idx = getTempoClass(period,min_period,max_period)
			tempo_curve[start_pos:end_pos] = period
			tempo_curve_dig[start_pos:end_pos,idx] = 1

		tempo_curve[0:locations[0]] = diff[0]
		tempo_curve[locations[-1]:-1] = diff[-1]
		tempo_curve[-1] = diff[-1]

		idx = getTempoClass(diff[0], min_period, max_period)
		tempo_curve_dig[0:locations[0], idx] = 1

		idx = getTempoClass(diff[-1], min_period, max_period)
		tempo_curve_dig[locations[-1]:-1, idx] = 1
		tempo_curve_dig[-1, idx] = 1

	return tempo_curve, tempo_curve_dig
