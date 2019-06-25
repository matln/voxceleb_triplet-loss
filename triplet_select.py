from itertools import combinations, permutations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


class AllTripletSelector():
	"""
    Returns all possible triplets
    May be impractical in most cases
	"""
	
	def __init__(self):
		pass
	
	def get_triplets(self, embeddings, labels):
		labels = labels.cpu().data.numpy()
		triplets = []
		for label in set(labels):
			label_mask = (labels == label)
			label_indices = np.where(label_mask)[0]
			if len(label_indices) < 2:
				continue
			negative_indices = np.where(np.logical_not(label_mask))[0]
			# All anchor-positive pairs
			anchor_positives = list(combinations(label_indices, 2))
			
			# Add all negatives for all positive pairs
			temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
			                 for neg_ind in negative_indices]
			triplets += temp_triplets
		
		return torch.LongTensor(np.array(triplets))

def batch_all(loss_values):
	hard_negatives = np.where(loss_values > 0)[0]
	return list(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
	hard_negative = np.argmax(loss_values)
	return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
	hard_negatives = np.where(loss_values > 0)[0]
	return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
	semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
	# return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None
	return semihard_negatives[np.argmax(loss_values[semihard_negatives])] if len(semihard_negatives) > 0 else None
	

class FunctionNegativeTripletSelector():
	"""
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
	"""
	
	def __init__(self, margin, negative_selection_fn, all_positive=True, cpu=True, squared=True):
		self.cpu = cpu
		self.margin = margin
		self.negative_selection_fn = negative_selection_fn
		self.all_positive = all_positive
		self.squared=squared
	
	def pdist(self, vectors):
		distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(
			2).sum(dim=1).view(-1, 1)
		# 未知原因，主对角线会出现负数，例如-4.3e-7，之类的很小的负数
		# 简单粗暴的把对角线都置为0
		index = [i for i in range(len(distance_matrix))]
		distance_matrix[index, index] = 0
		# distance_matrix += torch.eye(distance_matrix.shape[0]).to(distance_matrix.device) * 1e-8
		dist = distance_matrix.sqrt()
		
		return dist

	def pdist_cos(self, vectors):
		B, D = vectors.size()
		dot = vectors @ vectors.t()
		norm1 = vectors.norm(dim=1)
		norm2 = vectors.norm(dim=1).view(1, B).t()
		dot /= norm1 * norm2
		return dot.t()
	
	def get_triplets(self, embeddings, labels):
		if self.cpu:
			embeddings = embeddings.cpu()
		# TODO: cos or l2
		distance_matrix = euclidean_distances(embeddings.numpy(), squared=self.squared)
		# distance_matrix = self.pdist_cos(embeddings))
		# distance_matrix = self.pdist(embeddings)
		
		# for ii in range(len(distance_matrix)):
		# 	assert np.isnan(distance_matrix[ii, ii]) == False, 'dist_matrix:{}\n\n{}\n\n{}'.format(distance_matrix, ceshi, embeddings[ii])
	
		labels = labels.cpu().data.numpy()
		triplets = []
		
		for label in set(labels):
			label_mask = (labels == label)
			label_indices = np.where(label_mask)[0]
			if len(label_indices) < 2:
				continue
			negative_indices = np.where(np.logical_not(label_mask))[0]
			anchor_positives = list(permutations(label_indices, 2))  # All anchor-positive pairs
			anchor_positives = np.array(anchor_positives)
			ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
			
			if not self.all_positive:
				## batch hard strategy
				# select hardest anchor-positive and hardest anchor-negative tripelts
				anchor_positives = []
				ap_distances = []
				for anchor in label_indices:
					# If use cos similarity, where max -> min, label_indices need to delete anchor, otherwise
					# the minimum of ap is (anchor, anchor)
					# TODO: cos or l2
					ap_distances.append(max(distance_matrix[anchor.repeat(len(label_indices)), label_indices]))
					hardest_ap_idx = np.argmax(
						distance_matrix[anchor.repeat(len(label_indices)), label_indices])
					anchor_positives.append([anchor, label_indices[hardest_ap_idx]])
			
			for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
				# TODO: cos or l2
				loss_values = ap_distance - distance_matrix[np.array([anchor_positive[0]]),
				                                            negative_indices] + self.margin
				# loss_values = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])),
				#                               torch.LongTensor(negative_indices)] - ap_distance + self.margin
				# loss_values = loss_values.cpu().numpy()
				
				hard_negative = self.negative_selection_fn(loss_values)

				if hard_negative is not None:
					if isinstance(hard_negative, list):
						for i in range(len(hard_negative)):
							_hard_negative = negative_indices[hard_negative[i]]
							triplets.append([anchor_positive[0], anchor_positive[1], _hard_negative])
					else:
						hard_negative = negative_indices[hard_negative]
						triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
				
		
		if len(triplets) == 0:
			triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
		
		triplets = np.array(triplets)
		
		return torch.LongTensor(triplets)


def BatchAllTripletSelector(margin, cpu=True, squared=True):
	return FunctionNegativeTripletSelector(margin=margin,
	                                       negative_selection_fn=batch_all,
	                                       cpu=cpu, squared=squared)


def HardestNegativeTripletSelector(margin, all_positive=True, cpu=True, squared=True):
	return FunctionNegativeTripletSelector(margin=margin,
	                                       negative_selection_fn=hardest_negative,
	                                       all_positive=all_positive,
	                                       cpu=cpu, squared=squared)


def RandomNegativeTripletSelector(margin, cpu=False, squared=True):
	return FunctionNegativeTripletSelector(margin=margin,
	                                       negative_selection_fn=random_hard_negative,
	                                       cpu=cpu, squared=squared)


def SemihardNegativeTripletSelector(margin, cpu=True, squared=True):
	return FunctionNegativeTripletSelector(margin=margin,
	                                       negative_selection_fn=lambda x: semihard_negative(x, margin),
	                                       cpu=cpu, squared=squared)



class DistanceWeightedSampling(nn.Module):
	'''
    parameters
    ----------
    batch_k: int
        number of images per class
    Inputs:
        data: input tensor with shapeee (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx
	'''

	def __init__(self, n_samples, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize =False,  **kwargs):
		super(DistanceWeightedSampling,self).__init__()
		self.n_samples = n_samples
		self.cutoff = cutoff
		self.nonzero_loss_cutoff = nonzero_loss_cutoff
		self.normalize = normalize
	
	def get_distance(self, x):
		_x = x.detach()
		sim = torch.matmul(_x, _x.t())
		dist = 2 - 2 * sim
		dist += torch.eye(dist.shape[0]).to(dist.device)  # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
		dist = dist.sqrt()
		# distance_matrix = -2 * _x.mm(torch.t(_x)) + _x.pow(2).sum(dim=1).view(1, -1) + _x.pow(
		# 	2).sum(dim=1).view(-1, 1)
		# distance_matrix += torch.eye(distance_matrix.shape[0]).to(distance_matrix.device) * 0.01
		# dist = distance_matrix.sqrt()
		return dist

	def get_triplets(self, embeddings, labels=None):
		k = self.n_samples
		n, d = embeddings.shape
		distance = self.get_distance(embeddings)
		distance = distance.clamp(min=self.cutoff)
		log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(torch.clamp(1.0 - 0.25*(distance*distance), min=1e-8)))

		if self.normalize:
			log_weights = (log_weights - log_weights.min()) / (log_weights.max() - log_weights.min() + 1e-8)

		weights = torch.exp(log_weights - torch.max(log_weights))

		if embeddings.device != weights.device:
			weights = weights.to(embeddings.device)

		mask = torch.ones_like(weights)
		for i in range(0,n,k):
			mask[i:i+k, i:i+k] = 0

		mask_uniform_probs = mask.double() *(1.0/(n-k))

		weights = weights * mask * ((distance < self.nonzero_loss_cutoff).float())
		weights_sum = torch.sum(weights, dim=1, keepdim=True)
		_weights = weights / (weights_sum + 1e-8)

		# a_indices = []
		# p_indices = []
		# n_indices = []
		triplets = []

		np_weights = _weights.cpu().numpy()
		mask_uniform_probs = mask_uniform_probs.cpu().numpy()
		_max = []
		_min = []
		
		for i in range(n):
			block_idx = i // k
			
			for value in np_weights[i]:
				assert np.isnan(value) == False, 'i:{}\n\n\n\n' \
				                                 'np_weights[i]: {}\n\n\n\n\n' \
				                                 'distance: {}\n\n\n\n' \
				                                 'weights_sum: {}\n\n\n\n' \
				                                 'weights[i]:{}\n\n\n\n' \
				                                 'log_weights[i]: {}'.format(i,
				                                                          np_weights[i],
				                                                          distance,
				                                                          weights_sum,
				                                                          weights[i],
				                                                          log_weights[i])
				
			_max.append(max(np_weights[i]))
			_min.append(min(set(np_weights[i]) - set(np_weights[i][block_idx * k:(block_idx + 1) * k])))
				
			if weights_sum[i] != 0:
					n_indices = np.random.choice(n, k-1, p=np_weights[i]).tolist()
				
			else:
				n_indices = np.random.choice(n, k-1, p=mask_uniform_probs[i]).tolist()
			idx = 0
			for j in range(block_idx * k, (block_idx + 1) * k):
				if j != i:
					triplets.append([i, j, n_indices[idx]])
					idx += 1
		
		# print(_min, '\n', _max)
		# print(np.mean(np.array(_min)))
		# print(np.mean(np.array(_max)))
		
		triplets = np.array(triplets)
	
		return torch.LongTensor(triplets)