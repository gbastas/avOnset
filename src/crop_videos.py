import cv2
import librosa
import numpy as np
import sys
import argparse
import pretty_midi
import matplotlib.pyplot as plt
import os
import shutil


def crop_video_new(path_to_video_file, n_pieces, path_to_subdir, AudioDataSep):
	filename = path_to_video_file.split('/')[-1]
	perfID = int(filename.split('_')[1])
	vidcap = cv2.VideoCapture(path_to_video_file)
	success,image = vidcap.read()
	height, width, layers = image.shape
	print(width, height)
	# Delete old cropped videos
	try:
		os.system('rm '+path_to_subdir+'/*.mkv')
	except:
		print()
	
	for i in range(n_pieces):
		shift = 60

		crop = str(width//n_pieces)+":"+str(height)+":"+str(i*width//n_pieces - i//(i-0.00001)*shift)+":"+str(0) # NOTE: Set the crop-window. Slide it a bit to the left for non-zero i: "- i//(i-0.00001)*60"
		print(crop)
		os.system('ffmpeg -vb 11M -i '+"'"+path_to_video_file+"'"+' -filter:v "crop='+crop+'" -c:a copy out'+str(i+1)+'.mp4')
		os.system('ffmpeg -i out'+str(i+1)+'.mp4 -c copy -an mute-out'+str(i+1)+'.mp4')

		vid_name = "VidSep_"+AudioDataSep[i].split('/')[-1].split('_',1)[1].split('.')[0]+'.mkv' # e.g. 
		# print("AAAAAAA", vid_name)

		os.system('ffmpeg -i mute-out'+str(i+1)+'.mp4 -i '+"'"+AudioDataSep[i]+"'"+' -c copy '+path_to_subdir+'/'+vid_name)
		os.system('rm out'+str(i+1)+'.mp4')
		os.system('rm mute-out'+str(i+1)+'.mp4')


def main(parameters):

	path_to_data = parameters['pathToData']
	for subdir in os.listdir(path_to_data):
		try:
			# load the video files 
			AudioDataSep=[]
			for filename in os.listdir(path_to_data+subdir):
				if not(filename.startswith('.')) and ('Vid_' in filename) and filename.endswith(".mp4"): 
					print('MixΜidi: '+filename)
					video_file_mix = filename
					path_to_video_file = path_to_data+subdir+'/'+filename
					n_instruments = len( video_file_mix.split('_') ) - 3 # e.g. Vid_05_Entertainer_tpt_tpt.mp4 -> 2 intruments
				elif not(filename.startswith('.')) and ('AuSep' in filename):
					print('SepAudio: '+filename)
					path_to_wav_file = path_to_data+subdir+'/'+filename
					AudioDataSep.append( path_to_wav_file )

			crop_video_new(path_to_video_file, n_instruments, '"'+path_to_data+subdir+'"', AudioDataSep)

		except  NotADirectoryError as e:
			# print('ERROR:', type(e).__name__, e)
			continue


if __name__ == '__main__':
	# Set up command-line argument parsing
	parser = argparse.ArgumentParser(
		description='Crop videos. E.g. yes | python src/crop_videos.py --pathToData "/media/gbastas/New Volume/users/grigoris/Datasets/uc3/Dataset/" && notify-send End "process finished"',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--pathToData', help='Path to the dataset')
	parameters = vars(parser.parse_args(sys.argv[1:]))

	main(parameters)

