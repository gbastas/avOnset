import numpy as np
import os
import json
import cv2


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