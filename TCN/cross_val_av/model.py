from torch import nn
import sys
# from CompactBilinearPooling import CompactBilinearPooling
sys.path.append("../")
from tcn import TemporalConvNet#, SimpleConvNet
import torch.nn.functional as F
import torch

# from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class TCN_Vis_fusion(nn.Module):
	def __init__(self, args, A_input_size, B_input_size, A_model_name, B_model_name, output_size, num_A_channels, num_B_channels, kernel_size, dropout, dilations):
		super(TCN_Vis_fusion, self).__init__()

		self.multiLoss = args.multiLoss
		self.fusion_strat = args.fusion_strat

		# num_A_channels = num_A_channels[:-3]
		# num_B_channels = num_B_channels[:-3]

		self.tcn_a = TemporalConvNet(A_input_size, num_A_channels, kernel_size, dropout=dropout)
		self.tcn_b = TemporalConvNet(B_input_size, num_B_channels, kernel_size, dropout=dropout)

		#NOTE:
		self.linear = nn.Linear(num_A_channels[-1]+num_B_channels[-1], output_size)
		self.linearInd = nn.Linear(num_A_channels[-1], output_size)
		# self.linear = nn.Linear( num_A_channels[-1], output_size)

		self.tcn_fuse = TemporalConvNet(num_A_channels[-1]+num_B_channels[-1], [2*n for n in num_A_channels[:3]], kernel_size, dropout=dropout) # NOTE: A or B here [2*n for n in num_A_channels]????
		# self.tcn_fuse = TemporalConvNet(num_A_channels[-1]+num_B_channels[-1], [num_A_channels[-1]*2, num_A_channels[-1], num_A_channels[-1]], kernel_size, dropout=dropout) # NOTE: A or B here [2*n for n in num_A_channels]????
		self.softmax = nn.Softmax(dim=2)
		


	def forward(self, x, y):
		# x needs to have dimension (N, C, L) in order to be passed into CNN
		output_a = self.tcn_a(x.transpose(1, 2)).transpose(1, 2)
		output_b = self.tcn_b(y.transpose(1, 2)).transpose(1, 2)
		output = torch.cat((output_a, output_b), 2) # or 2

		if self.fusion_strat=='tcn':
			output = self.tcn_fuse(output.transpose(1, 2)).transpose(1, 2)
		output = self.linear(output).double()

		if self.multiLoss:
			# Extra for multiple loss propagation
			output_a = self.linearInd(output_a).double()
			output_b = self.linearInd(output_b).double()

			# return (self.softmax(output) + self.softmax(output_a) + self.softmax(output_b))/3
			return (self.softmax(output), self.softmax(output_a), self.softmax(output_b))

		else:
			return self.softmax(output)

class TCN_Pix_Skltn_fusion(nn.Module):
	def __init__(self, args, A_input_size, A_model_name, B_model_name, output_size, num_A_channels, num_B_channels, kernel_size, dropout, dilations):
		super(TCN_Pix_Skltn_fusion, self).__init__()

		self.multiLoss = args.multiLoss
		self.fusion_strat = args.fusion_strat

		self.tcn = TemporalConvNet(A_input_size, num_A_channels, kernel_size, dropout=dropout)

		self.cnn = CNN()
		self.cnn_tcn = TemporalConvNet(8*4*4, num_channels=num_B_channels, kernel_size=kernel_size, dropout=dropout) 

		self.linear = nn.Linear(num_A_channels[-1]+num_B_channels[-1], output_size)
		self.linearInd = nn.Linear(num_A_channels[-1], output_size)

		self.tcn_fuse = TemporalConvNet(num_A_channels[-1]+num_B_channels[-1], [2*n for n in num_A_channels[:3]], kernel_size, dropout=dropout) # NOTE: A or B here [2*n for n in num_A_channels]????
		self.softmax = nn.Softmax(dim=2)
		
	def forward(self, x, y):
		# x needs to have dimension (N, C, L) in order to be passed into CNN

		# y = torch.cat([y, y[-1].unsqueeze(0)], dim=0)

		imgs = y.transpose(1,3)	
		imgs = imgs/255.0
		cnn_out = self.cnn(imgs)
		cnn_out = cnn_out.squeeze(0)
		cnn_out = cnn_out.reshape(cnn_out.shape[0], -1) # make it one-dimensional
		cnn_out = cnn_out.unsqueeze(0)

		output_a = self.tcn(x.transpose(1, 2)).transpose(1, 2)
		output_b = self.cnn_tcn(cnn_out.transpose(1, 2)).transpose(1, 2)

		# print('output_a:', output_a.size(), 'output_b:', output_b.size())
		# print('x:', x.size(), 'y:', y.size())

		output = torch.cat((output_a, output_b), 2) # or 2

		if self.fusion_strat=='tcn':
			output = self.tcn_fuse(output.transpose(1, 2)).transpose(1, 2)
		output = self.linear(output).double()

		if self.multiLoss:
			# Extra for multiple loss propagation
			output_a = self.linearInd(output_a).double()
			output_b = self.linearInd(output_b).double()

			# return (self.softmax(output) + self.softmax(output_a) + self.softmax(output_b))/3
			return (self.softmax(output), self.softmax(output_a), self.softmax(output_b))

		else:
			return self.softmax(output)



class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		cnn_dropout = 0.0

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=1), 
			# nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2), 
			nn.BatchNorm2d(4),
			nn.ReLU(),
			nn.Dropout(cnn_dropout),
			nn.MaxPool2d(kernel_size=3, stride=3))
		self.layer2 = nn.Sequential(
			nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=1),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.Dropout(cnn_dropout),
			nn.MaxPool2d(kernel_size=3, stride=3)) 
		self.layer3 = nn.Sequential(
			nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=1),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.Dropout(cnn_dropout),
			nn.MaxPool2d(kernel_size=2, stride=2))
			# nn.MaxPool2d(kernel_size=3, stride=3))


# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN, self).__init__()
# 		cnn_dropout = 0.0

# 		# self.downsample1 = nn.Sequential(
# 		# 	nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0), 
# 		# 	nn.BatchNorm2d(4),
# 		# 	nn.MaxPool2d(kernel_size=3, stride=3))
# 		# self.downsample2 = nn.Sequential(
# 		# 	nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0), 
# 		# 	nn.BatchNorm2d(8),
# 		# 	nn.MaxPool2d(kernel_size=3, stride=3))
# 		# self.downsample3 = nn.Sequential(
# 		# 	nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0), 
# 		# 	nn.BatchNorm2d(8),
# 		# 	nn.MaxPool2d(kernel_size=3, stride=3))
# 		# self.downsample4 = nn.Sequential(
# 		# 	nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
# 		# 	nn.BatchNorm2d(8),
# 		# 	nn.MaxPool2d(kernel_size=2, stride=2))

# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2), 
# 			nn.BatchNorm2d(4),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			# nn.MaxPool2d(kernel_size=3, stride=3))
# 			nn.MaxPool2d(kernel_size=4, stride=4))
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=4, stride=4))
# 		self.layer3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=4, stride=4))
# 		# self.layer4 = nn.Sequential(
# 		# 	nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
# 		# 	nn.BatchNorm2d(8),
# 		# 	nn.ReLU(),
# 		# 	nn.Dropout(cnn_dropout),
# 		# 	nn.MaxPool2d(kernel_size=2, stride=2))


# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN, self).__init__()
# 		cnn_dropout = 0.0
# 		self.downsample1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(4),
# 			nn.MaxPool2d(kernel_size=3, stride=3))		
# 		self.downsample2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=3, stride=3))
# 		self.downsample3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=3, stride=3))

# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1), 
# 			nn.BatchNorm2d(4),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=3, stride=3))
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=3, stride=3)) 
# 		self.layer3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=3, stride=3))

# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN, self).__init__()
# 		cnn_dropout = 0.0
# 		self.downsample1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(4),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.downsample2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.downsample3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		

# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=1), 
# 			nn.BatchNorm2d(4),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.layer3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		


# class CNN(nn.Module):
# 	def __init__(self):
# 		super(CNN, self).__init__()
# 		cnn_dropout = 0.0
# 		self.downsample1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(4),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.downsample2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.downsample3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0), 
# 			nn.BatchNorm2d(8),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		

# 		self.layer1 = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2), 
# 			nn.BatchNorm2d(4),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.layer2 = nn.Sequential(
# 			nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		
# 		self.layer3 = nn.Sequential(
# 			nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2),
# 			nn.BatchNorm2d(8),
# 			nn.ReLU(),
# 			nn.Dropout(cnn_dropout),
# 			nn.MaxPool2d(kernel_size=2, stride=2))		

	def forward(self, x):
		# residual = self.downsample1(x)
		out = self.layer1(x)
		# out += residual

		# residual = self.downsample2(out)
		out = self.layer2(out)
		# out += residual

		# residual = self.downsample3(out)
		out = self.layer3(out)
		# out += residual

		# residual = self.downsample4(out)
		# out = self.layer4(out)
		# out += residual

		return out

class CNN_TCN(nn.Module): # pixel input
	def __init__(self, args, output_size, num_channels, kernel_size, dropout):
		super(CNN_TCN, self).__init__()
		self.step =args.cnn_step

		self.cnn = CNN()

		# self.tcn = TemporalConvNet(8*3*3, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)  
		self.tcn = TemporalConvNet(8*4*4, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)  
		# self.tcn = TemporalConvNet(8*5*5, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)  
		# self.tcn = TemporalConvNet(16*4*4, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)  
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.softmax = nn.Softmax(dim=2)


	# def forward(self, x, lengths): # NOTE: lengths variable is there on purpose even if it is not used
	def forward(self, x): 

		part_seq = []
		step = self.step
		n=0
		for i in range(0, len(x), step):
			if i+step+n<len(x):
				xs = x[i:(i+step+n),:,:,:].cuda()
			else:
				xs = x[i:len(x),:,:,:].cuda()

			xs = xs/255.0 # NOTE		

			imgs = xs.transpose(1, 3) # h, w, 3 -> 3, w, h
			out = self.cnn(imgs)

			out = out.squeeze(0)
			out = out.reshape(out.shape[0], -1) # make it one-dimensional

			# print('out: ', out.size())

			part_seq += out.cpu().detach()

		out_seq = torch.stack(part_seq).cuda()

		out_seq = out_seq.unsqueeze(0)
		out_seq = self.tcn(out_seq.transpose(1, 2)).transpose(1, 2)
		out_seq = self.linear(out_seq).double()
		return self.softmax(out_seq)	


		# x = x.cuda()
		# # x = torch.cat([x, x[-1].unsqueeze(0)], dim=0)
		# # img1 = x[1:].transpose(1,3)
		# # img2 = x[:-1].transpose(1,3)	
		# # img = img1-img2

		# imgs = x.transpose(1,3)	

		# imgs = imgs/255.0

		# out = self.cnn(imgs)

		# out = out.squeeze(0)
		# out = out.reshape(out.shape[0], -1) # make it one-dimensional

		# # print('out: ', out.size())

		# out_seq = out
		# # part_seq += out.cpu().detach()
		# # out_seq = torch.stack(part_seq).cuda()

		# out_seq = out_seq.unsqueeze(0)
		# out_seq = self.tcn(out_seq.transpose(1, 2)).transpose(1, 2)
		# out_seq = self.linear(out_seq).double()
		# return self.softmax(out_seq)	



class TCN(nn.Module):
	def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilations):
		super(TCN, self).__init__()
		# print("input_size", input_size)
		if dilations:
			# self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
			self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
		else:
			self.tcn = SimpleConvNet(input_size, num_channels, kernel_size, dropout=dropout)

		# self.mask = TCNmask(input_size, output_size, num_channels, kernel_size, dropout, dilations)
		# self.conv1x1 = nn.Conv1d(in_channels=num_channels[-1], out_channels=num_channels[-1], kernel_size=1, stride=1, padding=0)
		# self.conv1x1_2 = nn.Conv1d(in_channels=num_channels[-1], out_channels=output_size, kernel_size=1, stride=1, padding=0, dilation=1)
		# self.relu = nn.ReLU()

		self.linear = nn.Linear(num_channels[-1], output_size)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, x):
		# x needs to have dimension (N, C, L) in order to be passed into CNN
		output = self.tcn(x.transpose(1, 2)).transpose(1, 2)

		output = self.linear(output).double()
		return self.softmax(output)


class TCNmask(nn.Module):
	def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilations):
		super(TCNmask, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

	def forward(self, x):
		# x needs to have dimension (N, C, L) in order to be passed into CNN
		output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
		output = torch.sigmoid(output)
		# return self.softmax(output)
		return output


################## PRETRAINED #######################

class TCN_pre(nn.Module):
	def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilations):
		super(TCN_pre, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
		# self.linear = nn.Linear(num_channels[-1], output_size)
		# self.softmax = nn.Softmax(dim=2)        
	def forward(self, x):
		output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
		return output


class CNN_TCN_pre(nn.Module):
	def __init__(self, args, output_size, num_channels, kernel_size, dropout):
		super(CNN_TCN_pre, self).__init__()
		self.step = args.cnn_step
		self.cnn = CNN()
		self.tcn = TemporalConvNet(8*4*4, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)  

	def forward(self, x): 
		part_seq = []
		step = self.step
		n=0
		for i in range(0, len(x), step):
			if i+step+n<len(x):
				xs = x[i:(i+step+n),:,:,:].cuda()
			else:
				xs = x[i:len(x),:,:,:].cuda()

			xs = xs/255.0 # NOTE		

			imgs = xs.transpose(1, 3) # h, w, 3 -> 3, w, h
			out = self.cnn(imgs)

			out = out.squeeze(0)
			out = out.reshape(out.shape[0], -1) # make it one-dimensional

			# print('out: ', out.size())

			part_seq += out.cpu().detach()

		out_seq = torch.stack(part_seq).cuda()

		out_seq = out_seq.unsqueeze(0)
		out_seq = self.tcn(out_seq.transpose(1, 2)).transpose(1, 2)

		return out_seq

class TCN_fusion(nn.Module):
	def __init__(self, args, A_input_size, B_input_size, A_model_name, B_model_name, output_size, num_A_channels, num_B_channels, kernel_size, dropout, dilations):
		super(TCN_fusion, self).__init__()
		self.fusion_strat = args.fusion_strat
		self.multiTest = args.multiTest
		self.multiLoss = args.multiLoss
		
		model = TCN_pre(A_input_size, output_size, num_A_channels, kernel_size, dropout, dilations)

		self.tcn_A = self.discard_out_layer(model, torch.load(open(A_model_name, "rb")).state_dict() )
		# self.tcn_A_tutti = torch.load(open(A_model_name, "rb"))

		if args.modality == 'Body-Hand':
			model = CNN_TCN_pre(args, output_size, num_B_channels, kernel_size, dropout)
		else:
			model = TCN_pre(B_input_size, output_size, num_B_channels, kernel_size, dropout, dilations)

		self.tcn_B = self.discard_out_layer(model, torch.load(open(B_model_name, "rb")).state_dict() )
		# self.tcn_B_tutti = torch.load(open(B_model_name, "rb"))

		if args.freeze:
			for p in self.tcn_B.parameters(): p.requires_grad = False
			for p in self.tcn_A.parameters(): p.requires_grad = False

		# self.gate_tanh = torch.nn.Tanh()
		# self.gate_sigmoid = torch.nn.Sigmoid()

		self.linear = nn.Linear(num_B_channels[-1]*2, output_size)

		# self.linearInd = nn.Linear(num_B_channels[-1], output_size)
		# self.linearIndA = self.tcn_A_tutti.linear
		# self.linearIndB = self.tcn_B_tutti.linear

		# self.cbplayer = CompactBilinearPooling(num_A_channels[-1], num_B_channels[-1], 8000)
		# self.cbplayer = CompactBilinearPooling(num_B_channels[-1], num_B_channels[-1], 512)#.cuda()
		self.lstm = nn.LSTM(input_size = 512, hidden_size = 512, num_layers=2) 
		self.hidden_cell = (torch.zeros(1,1,512), 
							torch.zeros(1,1,512))		
		self.linear2 = nn.Linear(512, output_size)


		self.tcn_fuse = TemporalConvNet(num_B_channels[-1]*2, [2*n for n in num_B_channels[:3]], kernel_size, dropout=dropout)
		self.softmax = nn.Softmax(dim=2)


	def discard_out_layer(self, model, model_dict):
		# 1. filter out unnecessary keys
		# print(model_dict.items())
		new_dict = {k: v for k, v in model_dict.items() if k not in ['linear.weight', 'linear.bias'] }
		# 2. overwrite entries in the existing state dict
		# model_dict.update(new_dict) 
		# 3. load the new state dict
		model.load_state_dict(new_dict)
		# model.load_state_dict(model_dict)
		return model


	def forward(self, A, B):
		# x needs to have dimension (N, C, L) in order to be passed into CNN
		output_A = self.tcn_A(A)
		output_B = self.tcn_B(B)
		output = torch.cat((output_A, output_B), 2) # or 2
		# output = torch.mul(output_A, output_B) # or 2

		# # NOTE: PixelCNN gate
		# gated_tanh = self.gate_tanh(output)
		# gated_sigmoid = self.gate_sigmoid(output)
		# output = gated_tanh * gated_sigmoid

		# if self.cbpbool:
		# 	output = self.cbplayer()

		if self.fusion_strat=='tcn':
			output = self.tcn_fuse(output.transpose(1, 2)).transpose(1, 2)
			output = self.linear(output).double()
		if self.fusion_strat=='lstm':
			# output = self.cbplayer(output_A, output_B)
			output_A, output_B = output_A.cpu().detach(), output_B.cpu().detach()
			output, self.hidden_cell = self.lstm(output)
			output = self.linear2(output).double()
		else:
			output = self.linear(output).double()

		# self.multiLoss = True	# NOTE: maybe need to uncomment for testing	
		if self.multiTest:
			# print('multiTest!!!!!!!')
			# Extra for multiple loss propagation
			output_A = self.tcn_A_tutti(A)
			output_B = self.tcn_B_tutti(B)

			# output_A = self.linearIndA(output_A).double()
			# output_B = self.linearIndB(output_B).double()
			# return (self.softmax(output), self.softmax(output_A), self.softmax(output_B))
			return (self.softmax(output), output_A, output_B)

		elif self.multiLoss:
			output_A = self.linearIndA(output_A).double()
			output_B = self.linearIndB(output_B).double()
			return (self.softmax(output), self.softmax(output_A), self.softmax(output_B))
		else:
			return self.softmax(output)
