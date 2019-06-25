import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import transforms
from dataset import TrainDataset, ToTensor_test, ToTensor, normalize_frames, TestDatasetLMDB, zero_padding_test,\
	TruncatedInput, OnlineBatchSampler, TestTruncatedInput, HalfSpec
from model import VGGVox1, OnlineTripletLoss
from triplet_select import DistanceWeightedSampling, HardestNegativeTripletSelector, \
	SemihardNegativeTripletSelector, BatchAllTripletSelector
from metric import compute_eer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# http://www.bubuko.com/infodetail-2977032.html
# https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823/4
# torch.multiprocessing.set_sharing_strategy('file_system')
# but easily cause shared memory leak:
# http://www.sohu.com/a/225270797_491081


# %% constants
import constants as c

# LMDB_test_path = '../data/vox1_spectrogram/test_buckets6900.lmdb'
LMDB_test_path = '../data/vox1_spectrogram/test.lmdb'
embedding_dim = 256    # you can change it to 256
batch_size = 128
num_workers = 8
n_classes = 60
n_samples = 10
pretrain_lr_init = 0.01
pretrain_lr_last = 0.0001
pretrain_epoch_num = 30
margin = 0.3
triplet_lr_init = 0.005
triplet_lr_last = 0.00005
triplet_epoch_num = 30

# %% dataset
transform = transforms.Compose([
	TruncatedInput(),
	HalfSpec(),
	normalize_frames(),
	ToTensor()
])

trainset = TrainDataset(transform=transform)
pretrainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True)

# -----------------------------------------------------------------------------------------------

iden_split_path = './voxceleb1_verification_trainset.txt'
_split = pd.read_table(iden_split_path, sep=' ', header=None, names=['label', 'path'])
_labels = _split['label']
batch_sampler = OnlineBatchSampler(_labels,  # .values: dataframe to ndarray
                                   n_classes=n_classes,
                                   n_samples=n_samples)
online_trainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                     batch_sampler=batch_sampler,
                                                     num_workers=num_workers)

# ----------------------------------------------------------------------------------------------

transform_test = transforms.Compose([
	normalize_frames(),
	zero_padding_test(max_width=1000),   # 6900 is the max width of test utterence spectrograms
	ToTensor_test()
])

testset = TestDatasetLMDB(LMDB_test_path, transform=transform_test)
testset_loader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=len(c.device_ids),
                                             num_workers=num_workers,
                                             shuffle=False)

# %% network
net = VGGVox1(num_classes=1211, emb_dim=embedding_dim)
net = nn.DataParallel(net, device_ids=c.device_ids)  # Multiple GPUs
net.to(c.device)

# %% define optimizer and learning rate scheduler
optimizer_triplet = optim.SGD(net.parameters(), lr=triplet_lr_init, momentum=c.MOMENTUM,
                        weight_decay=c.WEIGHT_DECAY)
gamma_triplet = 10 ** (np.log10(triplet_lr_last / triplet_lr_init) / (triplet_epoch_num - 1))
lr_scheduler_triplet = optim.lr_scheduler.StepLR(optimizer_triplet, step_size=1, gamma=gamma_triplet)

criterion_pretrain = nn.CrossEntropyLoss()
optimizer_pretrain = optim.SGD(net.parameters(), lr=pretrain_lr_init, momentum=c.MOMENTUM,
                           weight_decay=c.WEIGHT_DECAY)
gamma_pretrain = 10 ** (np.log10(pretrain_lr_last / pretrain_lr_init) / (pretrain_epoch_num - 1))
lr_scheduler_pretrain = optim.lr_scheduler.StepLR(optimizer_pretrain, step_size=1, gamma=gamma_pretrain)

# %% train and test
# pretrained_dict = torch.load('./data/checkpoint/pretrain_epoch_23')
# net.load_state_dict(pretrained_dict)

def checkpoint():
	# load checkpoint model
	for epoch in range(15, 20, 2):
		print('epoch:%d/%d' % (epoch, 30))
		pretrained_dict = torch.load('./data/vox1_model_save/VGG/pretrain_256/epoch_%d'%epoch)
		net.load_state_dict(pretrained_dict)
		evaluation_1()


def evaluation_1():
	# 由于网络中使用了全局平均池化层，所以可以输入任意帧数的语谱图(512×N)。
	# 测试时将整段测试语音输入，由于输入语谱图宽度不一致，所以无法输入一个mini-batch的数据，
	# 但是，为了加快速度(我的服务器可以使用6块GPU)，我们可以设置batch_size=6，由于6块GPU每块都有一个完整的
	# 网络(nn.DataParallel)，在每块GPU上分别计算一个可变宽度的语谱图，变相实现了每次迭代输入6个数据，一定程度上
	# 减少了计算时间。为了构造长度为6的input，预先设置max_width，我们需要对宽度不足max_width的语谱图进行补0(按列)，
	# 并在喂进网络之间还原出原始的语谱图，这就需要知道补0矩阵的列数，因此，我们在最后一列再添加一列关于补0列数
	# 的相关信息，所以最终一个batch数据的维度为(6, 1, 512, 1001)，(假设max_width=1000)
	
	def compute_similarity(spec1, spec2):
		spec1, spec2 = spec1.to(c.device), spec2.to(c.device)
		embedding1 = net(spec1, 'evaluation')
		embedding2 = net(spec2, 'evaluation')
		cos = torch.nn.CosineSimilarity(dim=1)
		similarity_cos = cos(embedding1, embedding2)
		
		dist_l2 = F.pairwise_distance(embedding1, embedding2)
		# distance to similarity
		similarity_l2 = -1 * dist_l2
		return similarity_cos, similarity_l2
	
	with torch.no_grad():
		net.eval()
		
		targets = []
		pred_similarity_l2 = []
		pred_similarity_cos = []
		
		for i, (target, spec1, spec2) in tqdm(enumerate(testset_loader)):
			if len(spec1) != len(c.device_ids):
				for j in range(len(spec1)):
					similarity_cos, similarity_l2 = compute_similarity(spec1[j:j + 1], spec2[j:j + 1])
					targets.append(target[j].item())
					pred_similarity_l2.extend(similarity_l2.cpu().numpy())
					pred_similarity_cos.extend(similarity_cos.cpu().numpy())
			else:
				similarity_cos, similarity_l2 = compute_similarity(spec1, spec2)
				targets.extend(target.numpy())
				pred_similarity_l2.extend(similarity_l2.cpu().numpy())
				pred_similarity_cos.extend(similarity_cos.cpu().numpy())
		EER1, minDCF1 = compute_eer(pred_similarity_l2, targets)
		EER2, minDCF2 = compute_eer(pred_similarity_cos, targets)
		print('test set:\n EER:%.2f  minDCF:%.2f \n EER:%.2f  minDCF:%.2f' % (EER1, minDCF1, EER2, minDCF2))


def pretrain():
	for epoch in range(pretrain_epoch_num):
		# ------------------------------------------------------------------------------------------- #
		# Pretrain
		# ------------------------------------------------------------------------------------------- #
		print('epoch:%d/%d' % (epoch + 1, pretrain_epoch_num))
		correct = 0
		total = 0
		running_loss = 0
		lr_scheduler_pretrain.step()
		# train
		net.train()  # batchnorm or dropout
		
		for i, (data, labels) in tqdm(enumerate(pretrainset_loader)):
			optimizer_pretrain.zero_grad()
			data = data.to(c.device)
			labels = labels.to(c.device)
			scores = net(data, 'pretrain')
			loss = criterion_pretrain(scores, labels)
			loss.backward()
			optimizer_pretrain.step()
			# Accuracy
			_, predicted = torch.max(scores.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			# Loss
			running_loss += float(loss)
			
			del labels, data, loss, scores
		print('Accuracy: %.2f, Loss: %.4f' % (100 * correct / total, running_loss / i))
		
		# ------------------------------------------------------------------------------------------------ #
		# Evaluation
		# ------------------------------------------------------------------------------------------------ #
		if epoch > 13 and epoch % 2 == 0:
			torch.save(net.state_dict(), './data/vox1_model_save/VGG/pretrain_256/epoch_%d' % (epoch + 1))
			evaluation_1()


def train_triplet():
	for epoch in range(triplet_epoch_num):
		# ------------------------------------------------------------------------------------------------- #
		# Train triplet loss
		# ------------------------------------------------------------------------------------------------- #
		print('epoch:%d/%d' % (epoch + 1, triplet_epoch_num))
		losses = []
		num_triplets = 0
		ap_dist_list = []
		an_dist_list = []
		lr_scheduler_triplet.step()
		net.train()  # batchnorm
		
		## triplet selector

		# Batch Hard
		# triplet_selector = HardestNegativeTripletSelector(margin=margin,
		#	                                                  all_positive=True, squared=True)  # use all anchor-positive pair or not
		# triplet_selector = BatchAllTripletSelector(margin=margin, squared=True)
		# triplet_selector = DistanceWeightedSampling(n_samples, normalize=True)
		triplet_selector = SemihardNegativeTripletSelector(margin=margin, squared=True)
		criterion_triplet = OnlineTripletLoss(margin=margin,
		                              triplet_selector=triplet_selector)
		
		for i, (data, labels) in tqdm(enumerate(online_trainset_loader)):
			optimizer_triplet.zero_grad()
			data = data.to(c.device)
			embeddings = net(data, 'triplet')
			loss, num, ap_dist, an_dist = criterion_triplet(embeddings, labels)
			loss.backward()
			optimizer_triplet.step()
			
			num_triplets += num
			losses.append(loss.item())
			ap_dist_list.append(ap_dist.item())
			an_dist_list.append(an_dist.item())
			torch.cuda.empty_cache()
		print('epoch:%d/%d, loss: %.4f, num_triplets: %d, ap_dist: %.4f, an_dist: %.4f \n' % (epoch + 1,
		                                                                                      triplet_epoch_num,
		                                                                                      np.mean(losses),
		                                                                                      num_triplets,
		                                                                                      np.mean(ap_dist_list),
		                                                                                      np.mean(an_dist_list)),
		                                                                                      ap_dist_list, '\n',
		                                                                                      an_dist_list)
		
		# ------------------------------------------------------------------------------------------------ #
		# Evaluation
		# ------------------------------------------------------------------------------------------------ #
		# Reference voxceleb2 4.4 evaluation method 1 or 3
		if epoch >= 5:
			torch.save(net.state_dict(), './data/vox1_model_save/VGG/triplets/epoch_%d' % (epoch + 1))
			evaluation_1()


if __name__ == '__main__':
	# pretrain()
	# train_triplet()
	checkpoint()

	# import time
	# # for num_workers in range(20, 50, 1):
	# pretrainset_loader = torch.utils.data.DataLoader(dataset=trainset,
	#                                                  batch_size=128,
	#                                                  num_workers=num_workers,
	#                                                  pin_memory=True,
	#                                                  shuffle=True)
	# start = time.time()
	# for i, (data, labels) in tqdm(enumerate(pretrainset_loader)):
	# 	pass
	# end = time.time()
	# print("Finish with: {} second num_worker={}".format(end-start, num_workers))