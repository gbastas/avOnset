'''
After having cropped each multi-instrument video (e.g. Vid_01_Jupiter_vn_vc.mp4 --> VidSep_1_vn_01_Jupiter.mkv & VidSep_2_vc_01_Jupiter.mkv) using ?src/crop_videos?,
we get the cropped videos stored in the standard dataset directory and create the corresponding set of json files containing all the data extracted through 
OpenPose for each frame (e.g. dirs VidSep_1_vn_01_Jupiter & VidSep_2_vc_01_Jupiter).

we run the script in the openpose dir like this
'''


# (av) $HOME\openpose>python ..\av_onset_2\src\run_multiple_openpose.py --pathToData "D:/users/grigoris/Datasets/uc3/Dataset/"  --poly True


import numpy as np
import argparse
import sys
import os
import shutil
import subprocess 
# pip install subprocess.run
# import librosa conda install -c conda-forge librosa

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def main(parameters):

	path_to_store = parameters['dirtostore'] #'./OpenPoseData/'
	# path_to_store = 'D:/users/grigoris/Datasets/'+parameters['dirtostore']

	path_to_data = parameters['pathToData']
	poly = parameters['poly']

	# Delete previusly created data
	try:
		shutil.rmtree(path_to_store)
	except:
		print('Error while deleting directory', path_to_store)
	os.mkdir(path_to_store)
	
	# Choose type_of_files_to_parse
	if poly:
		type_of_files_to_parse=".mp4"
	else:
		type_of_files_to_parse=".mkv"


	flag = False # NOTE: temp toDelete
	# Parse and extract skeletons weith openpose
	for subdir in os.listdir(path_to_data):
		if subdir.startswith('.'):
			continue
		# try:
		for filename in os.listdir(path_to_data+subdir):
			
			if filename.startswith('.'):
				continue

			if parameters['strings'] and (filename.split("_")[2] not in ['vn','vc','va','db']): 
				continue

			if parameters['no_strings'] and (filename.split("_")[2] in ['vn','vc','va','db']): 
				continue

			# TODO: TODELETE
			if filename == 'VidSep_1_fl_14_Waltz.mkv':
				flag=True		
			print(filename)
			if not flag:
				continue


			if filename.endswith(type_of_files_to_parse):
				path_to_src_file = path_to_data+subdir+'/'+filename
				print(path_to_src_file) 
				toRun = ["C:/Users/g.bastas/openpose/bin/OpenPoseDemo.exe","--video",path_to_src_file,
								"--write_json",path_to_store+filename.split('.')[0],
								"--write_video",path_to_store+filename.split('.')[0]+'.avi']

				if parameters['hands']:
					toRun += 	["--hand",
								"--hand_scale_number", '6', 
								"--hand_scale_range", '0.4',								
								"--hand_detector", '3']

				subprocess.run(toRun)

		

if __name__ == '__main__':
	# Set up command-line argument parsing
	parser = argparse.ArgumentParser(
		description='E.g. (av) python path/to/run_multiple_openpose.py --pathToData "D:/users/grigoris/Datasets/uc3/Dataset/"',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--pathToData', help='Path to the dataset')
	parser.add_argument('--dirtostore', default='OpenPoseData/')
	parser.add_argument('--poly', type=str2bool, help='Run for polyphonic (True) or monophonic (False) videos')
	parser.add_argument('--strings', type=str2bool, default=True, help='')
	parser.add_argument('--no_strings', type=str2bool, default=False, help='')
	# parser.add_argument('--keypoint_scale', type=int, default=1, help='0 (no normalization), 1 ([0,1]), 2 [(-1,1)]')
	parser.add_argument('--hands', type=str2bool, default=True, help='')

	parameters = vars(parser.parse_args(sys.argv[1:]))

	main(parameters)
