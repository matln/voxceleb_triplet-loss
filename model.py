import torch.nn as nn
import math
import torch
import constants as c
import torch.nn.functional as F

class ReLU(nn.Hardtanh):
	
	def __init__(self, inplace=False):
		super(ReLU, self).__init__(0, 20, inplace)
	
	def __repr__(self):
		inplace_str = 'inplace' if self.inplace else ''
		return self.__class__.__name__ + ' (' \
		       + inplace_str + ')'


class DeepSpeaker(nn.Module):
	
	def __init__(self, block, layers, embedding_size=512, num_classes=1211, feature_dim=64):
		super(DeepSpeaker, self).__init__()
		self.embedding_size = embedding_size
		
		self.inplanes = 64
		self.ResCNN_block1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
			nn.BatchNorm2d(64),
			ReLU(inplace=True),
			self._make_layer(block, 64, layers[0])
		)
		
		self.inplanes = 128
		self.ResCNN_block2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
			nn.BatchNorm2d(128),
			ReLU(inplace=True),
			self._make_layer(block, 128, layers[1])
		)
		
		self.inplanes = 256
		self.ResCNN_block3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
			nn.BatchNorm2d(256),
			ReLU(inplace=True),
			self._make_layer(block, 256, layers[2])
		)
		
		self.inplanes = 512
		self.ResCNN_block4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False),
			nn.BatchNorm2d(512),
			ReLU(inplace=True),
			self._make_layer(block, 512, layers[3])
		)
		
		if feature_dim == 64:
			self.fc = nn.Linear(512 * 4, self.embedding_size)
		elif feature_dim == 40:
			self.fc = nn.Linear(256 * 5, self.embedding_size)
		
		self.classifier_layer = nn.Linear(self.embedding_size, num_classes)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# TODO
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	
	def _make_layer(self, block, planes, blocks, stride=1):
		layers = []
		layers.append(block(self.inplanes, planes, stride))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)
	
	def forward_once(self, x):
		out = self.ResCNN_block1(x)
		out = self.ResCNN_block2(out)
		out = self.ResCNN_block3(out)
		out = self.ResCNN_block4(out)
		self.apool = nn.AvgPool2d((1, out.size(3)))
		out = self.apool(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out
	
	def forward(self, x, phase):
		out = self.forward_once(x)
		if phase == 'triplet':
			# TODO
			out = F.normalize(out, p=2, dim=1)
			# # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
			# alpha = 10
			# out = out * alpha
		elif phase == 'pretrain':
			out = self.classifier_layer(out)
		return out


class VGGVox2(nn.Module):
	
	def __init__(self, block, layers, num_classes=1211, embedding_size=512, alpha=10):
		super(VGGVox2, self).__init__()
		self.embedding_size = embedding_size
		self.num_classes=num_classes
		self.inplanes = 64
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
		
		self.avgpool1 = nn.AvgPool2d(kernel_size=(8, 3), stride=1)
		
		self.fc = nn.Sequential(
			nn.Conv2d(in_channels=512 * block.expansion, out_channels=512 * block.expansion, kernel_size=(16, 1)),
			nn.BatchNorm2d(num_features=512),
			nn.ReLU()
		)
		
		self.embedding_layer = nn.Linear(512 * block.expansion, self.embedding_size)
		
		self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		self.alpha.data.fill_(1)
		# self.alpha = alpha
		
		self.classifier_layer = nn.Linear(self.embedding_size, self.num_classes)
	
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)

	def forward_once(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.maxpool(out)
		
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)

		# out = self.avgpool1(out)
		## fc layer kernel_size: (9, 1) or (16, 1)
		out = self.fc(out)

		# Global average pooling layer
		_, _, _, width = out.size()
		self.avgpool2 = nn.AvgPool2d(kernel_size=(1, width))
		out = self.avgpool2(out)
		out = out.view(out.size(0), -1)
		out = self.embedding_layer(out)
		return out
	
	def forward(self, x, phase):
		if phase == 'evaluation':
			_padding_width = x[0, 0, 0, -1]
			out = x[:, :, :, :-1-int(_padding_width.item())]
			out = self.forward_once(out)
			out = F.normalize(out, p=2, dim=1)
			
		elif phase == 'triplet':
			out = self.forward_once(x)
			out = F.normalize(out, p=2, dim=1)
			
		elif phase == 'pretrain':
			out = self.forward_once(x)
			## Multiply by alpha as suggested in https://arxiv.org/pdf/1703.09507.pdf (L2-SoftMax)
			# out = F.normalize(out, p=2, dim=1)
			# out = out * self.alpha
			out = self.classifier_layer(out)
		else: return 'phase wrong!'
		return out


class VGGVox1(nn.Module):
	
	def __init__(self, num_classes=1211, emb_dim=1024):
		super(VGGVox1, self).__init__()
		self.num_classes = num_classes
		self.emb_dim = emb_dim
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1),
			nn.BatchNorm2d(num_features=96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True)
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True)
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
		)
		self.fc6 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9, 1)),
			nn.BatchNorm2d(num_features=4096),
			nn.ReLU(inplace=True)
		)
		self.fc7 = nn.Linear(in_features=4096, out_features=self.emb_dim)
		self.fc8 = nn.Linear(in_features=self.emb_dim, out_features=self.num_classes, bias=False)
	
	def forward_once(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.fc6(out)
		# global average pooling layer
		_, _, _, width = out.size()
		self.apool6 = nn.AvgPool2d(kernel_size=(1, width))
		out = self.apool6(out)
		out = out.view(out.size(0), -1)
		out = self.fc7(out)
		return out
	
	def forward(self, x, phase):
		if phase == 'evaluation':
			_padding_width = x[0, 0, 0, -1]
			out = x[:, :, :, :-1 - int(_padding_width.item())]
			out = self.forward_once(out)
			# out = F.normalize(out, p=2, dim=1)
		
		elif phase == 'triplet':
			out = self.forward_once(x)
			out = F.normalize(out, p=2, dim=1)
		
		elif phase == 'pretrain':
			out = self.forward_once(x)
			out = self.fc8(out)
		else:
			raise ValueError('phase wrong!')
		return out
	

class OnlineTripletLoss(nn.Module):
	"""
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    
    Reference: https://github.com/adambielski/siamese-triplet
	"""
	
	def __init__(self, margin, triplet_selector):
		super(OnlineTripletLoss, self).__init__()
		self.margin = margin
		self.triplet_selector = triplet_selector
	
	def forward(self, embeddings, target):
		triplets = self.triplet_selector.get_triplets(embeddings.detach(), target)
		
		if embeddings.is_cuda:
			triplets = triplets.to(c.device)
		
		# l2 distance
		ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
		an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		# ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).pow(.5)
		# an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).pow(.5)
		losses = F.relu(ap_distances - an_distances + self.margin)
		
		# # cosine similarity
		# cos = torch.nn.CosineSimilarity(dim=1)
		# ap_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
		# an_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
		# losses = F.relu(an_similarity - ap_similarity + self.margin)
		
		return losses.mean(), len(triplets), ap_distances.mean(), an_distances.mean()
	

class TripletLoss(nn.Module):
	"""
    Triplet loss
    Takes embeddings of an anchor sample, a posi7tive sample and a negative sample
	"""

	def __init__(self, margin):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, negative, size_average=True):
		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = F.relu(distance_positive - distance_negative + self.margin)
		return losses.mean() if size_average else losses.sum()