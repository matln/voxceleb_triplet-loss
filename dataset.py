from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
import pickle
import constants as c
import lmdb
import torch
import spectrogram as spec


class TrainDataset(Dataset):
	
	def __init__(self, transform=None):
		# 1. Initialize file paths or a list of file names.
		split_path = './voxceleb1_veri_dev.txt'
		split = pd.read_table(split_path, sep=' ', header=None, names=['label', 'path'])
		self.dataset = split['path']
		self.labels = split['label']
		self.transform = transform
	
	def __getitem__(self, item):
		# 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
		# 2. Preprocess the data (e.g. torchvision.Transform).
		# 3. Return a data pair (e.g. image and label).
		
		# label
		label = self.labels[item]
		
		# feature
		track_path = self.dataset[item]
		
		# TODO:
		# feature_path = os.path.join('../data/vox1_spectrogram/dev', track_path)
		# feature = np.load(feature_path.replace('.wav', '.npy'))
		audio_path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav', track_path)
		feature = spec.get_spectrum(audio_path)
		
		# transform
		if self.transform:
			feature = self.transform(feature)
		
		return feature, label
	
	def __len__(self):
		return len(self.dataset)


class VeriDatasetTest(Dataset):
	# Full length utterances are used in testing
	
	def __init__(self, transform=None):
		# 1. Initialize file paths or a list of file names.
		split_path = './voxceleb1_veri_test.txt'
		split = pd.read_table(split_path, sep=' ', header=None, names=['label', 'path'])
		self.dataset = split['path']
		self.transform = transform

	def __getitem__(self, item):
		wav_path = self.dataset.iloc[item]
		# wav_path_full = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav', wav_path)
		# feature = spec.get_spectrum(wav_path_full)
		
		fea_path_full = os.path.join('./data/feature_npy/test', wav_path)
		feature = np.load(fea_path_full.replace('.wav', '.npy'))

		# transforms
		if self.transform:
			feature = self.transform(feature)
		return wav_path, feature

	def __len__(self):
		return len(self.dataset)
	
	
class OnlineBatchSampler():
	"""
		samples n_classes and within this classes samples n_samples,
		return batches of size n_classes * n_samples
	"""
	
	def __init__(self, labels, n_classes, n_samples):
		self.labels = labels
		self.labels_set = list(set(self.labels))
		self.label_to_indices = {label: np.where(self.labels == label)[0]
		                         for label in self.labels_set}
		for label in self.labels_set:
			np.random.shuffle(self.label_to_indices[label])
		self.used_label_indice_count = {label: 0 for label in self.labels_set}
		self.count = 0
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dataset = len(self.labels)
		self.batch_size = self.n_classes * self.n_samples
	
	def __iter__(self):
		# generator
		self.count = 0
		while self.count + self.batch_size < self.n_dataset:
			classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
			indices = []
			for class_ in classes:
				indices.extend(self.label_to_indices[class_][
				               self.used_label_indice_count[class_]:
				               self.used_label_indice_count[class_] + self.n_samples])
				self.used_label_indice_count[class_] += self.n_samples
				if self.used_label_indice_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
					np.random.shuffle(self.label_to_indices[class_])
					self.used_label_indice_count[class_] = 0
			yield indices
			self.count += self.batch_size
	
	def __len__(self):
		return self.n_dataset // self.batch_size

# ------------------------------------------------------------------
# Transform

class normalize_frames(object):
	def __call__(self, m, epsilon=1e-12):
		mu = np.mean(m, 1, keepdims=True)
		std = np.std(m, 1, keepdims=True)
		return (m - mu) / (std + epsilon)
# return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


class TruncateInput(object):
	# Random select 3s segment from every utterance
	
	def __init__(self, num_frames=300):
		super(TruncateInput, self).__init__()
		self.num_frames = num_frames
	
	def __call__(self, frames_features):
		upper = frames_features.shape[1] - self.num_frames
		start = random.randint(0, upper)
		end = start + self.num_frames
		input_feature = frames_features[:, start:end]
		return np.array(input_feature)


class ToTensor(object):
	# Convert spectogram to tensor
	def __call__(self, spec):
		F, T = spec.shape
		# Now specs are of size (freq, time) and 2D but has to be 3D (channel dim)
		spec = spec.reshape(1, F, T)
		spec = spec.astype(np.float32)
		return torch.from_numpy(spec)


class TestTruncateInput(object):
	# Reference: https://github.com/a-nagrani/VGGVox/blob/master/test_getinput.m
	
	def __init__(self, max_width=6900):
		super(TestTruncateInput, self).__init__()
		# self.buckets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
		self.buckets = [i * 100 for i in range(1, int(max_width / 100) + 1)]
		
	def __call__(self, frames_feature):
		rsize = frames_feature.shape[1]
		for width in self.buckets:
			if width <= frames_feature.shape[1]:
				rsize = width
			else:
				break
		rstart = round((frames_feature.shape[1] - rsize) / 2)
		input_feature = frames_feature[:, rstart: rstart + rsize]
		
		return input_feature
	

class zero_padding_test(object):
	
	def __init__(self, max_width=1000):
		super(zero_padding_test, self).__init__()
		self.max_width = max_width
		
	def __call__(self, frames_feature):
		padding_width = self.max_width - frames_feature.shape[1]
		if frames_feature.shape[1] < self.max_width:
			zeros = np.zeros((frames_feature.shape[0], padding_width))
			frames_feature = np.concatenate((frames_feature, zeros), axis=1)
		padding_width_array = np.full((frames_feature.shape[0], 1), padding_width)
		frames_feature = np.concatenate((frames_feature, padding_width_array), axis=1)
		return frames_feature


# ----------------------------------------------------------------------
# lmdb dataset
# Reference: https://github.com/Lyken17/Efficient-PyTorch

def folder2lmdb(dpath, name='dev', write_frequency=5000):
	# convert to lmdb dataset
	dataset = TrainDataset(transform=None)
	data_loader = DataLoader(dataset, num_workers=32)
	
	lmdb_path = os.path.join(dpath, '%s.lmdb' % name)
	isdir = os.path.isdir(lmdb_path)
	
	print("Generate LMDB to %s" % lmdb_path)
	db = lmdb.open(lmdb_path, subdir=isdir, map_size=1099511627776,
	               meminit=False,
	               map_async=True)
	txn = db.begin(write=True)
	for i, (data, label) in tqdm(enumerate(data_loader)):
		txn.put(u'{}'.format(i).encode('ascii'), pickle.dumps((data.numpy(), label)))
		if i % write_frequency == 0:
			print("[%d/%d]" % (i, len(data_loader)))
			txn.commit()
			txn = db.begin(write=True)
	
	# finish iterating through dataset
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(i + 1)]
	with db.begin(write=True) as txn:
		txn.put(b'__keys__', pickle.dumps(keys))
		txn.put(b'__len__', pickle.dumps(len(keys)))
	
	print("Flushing database ...")
	db.sync()
	db.close()


class TrainDatasetLMDB(Dataset):
	def __init__(self, db_path, transform=None):
		super(TrainDatasetLMDB, self).__init__()
		self.db_path = db_path
		self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
		                     readonly=True, lock=False,
		                     readahead=False, meminit=False)
		with self.env.begin(write=False) as txn:
			self.length = pickle.loads(txn.get(b'__len__'))
			self.keys = pickle.loads(txn.get(b'__keys__'))
		
		self.transform = transform
	
	def __getitem__(self, item):
		env = self.env
		with env.begin(write=False) as txn:
			byteflow = txn.get(self.keys[item])
		unpacked = pickle.loads(byteflow)
		
		# load spec
		feature = unpacked[0][0]
		
		# load label
		label = unpacked[1][0]

		if self.transform is not None:
			feature = self.transform(feature)
		
		return feature, label
	
	def __len__(self):
		return self.length

	
if __name__ == '__main__':
	# folder2lmdb('./data/spectrogram')
	# folder2lmdb_test('../data/vox1_spectrogram')
	arr = np.array([[1, 2], [3, 4]])
	print(np.stack([arr] * 3, 2).shape)
	att1 = torch.randn(4, 4)
	arr1 = torch.ones(4, 4)
	print(att1)
	print(torch.where(att1>0, att1, arr1))
	
	
