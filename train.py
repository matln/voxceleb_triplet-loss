import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import transforms
import dataset as D
from backbone import VGGVox1, OnlineTripletLoss
from triplet_select import DistanceWeightedSampling, HardestNegativeTripletSelector, \
	SemihardNegativeTripletSelector, BatchAllTripletSelector
from metric import compute_eer
import spectrogram as spec
import os
import constants as c

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# http://www.bubuko.com/infodetail-2977032.html
# https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823/4
# torch.multiprocessing.set_sharing_strategy('file_system')
# but easily cause shared memory leak:
# http://www.sohu.com/a/225270797_491081

# %% dataset
# Pretrain
transform = transforms.Compose([
	D.TruncateInput(),
	D.normalize_frames(),
	D.ToTensor()
])

trainset = D.TrainDataset(transform=transform)
pretrainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                 batch_size=c.batch_size,
                                                 num_workers=c.num_workers,
                                                 pin_memory=True,
                                                 shuffle=True)

# ---------------------------------------
# Triplet loss

split_path = './voxceleb1_veri_dev.txt'
_split = pd.read_table(split_path, sep=' ', header=None, names=['label', 'path'])
_labels = _split['label']
batch_sampler = D.OnlineBatchSampler(_labels,  # .values: dataframe to ndarray
                                   n_classes=c.n_classes,
                                   n_samples=c.n_samples)
online_trainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                     batch_sampler=batch_sampler,
                                                     num_workers=c.num_workers)

# ----------------------------------------------------------------------------------------------
# Test

transform_test = transforms.Compose([
	D.normalize_frames(),
	D.zero_padding_test(max_width=7000),   # 7000 is the max width of test utterence spectrograms
	D.ToTensor()
])

testset = D.VeriDatasetTest(transform=transform_test)
testset_loader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=len(c.device_ids),
                                             num_workers=c.num_workers,
                                             pin_memory=True,
                                             shuffle=False)

# %% network
net = VGGVox1(num_classes=1211, emb_dim=c.embedding_dim)
net = nn.DataParallel(net, device_ids=c.device_ids)  # Multiple GPUs
net.to(c.device)

# %% define optimizer and learning rate scheduler
optimizer_triplet = optim.SGD(net.parameters(), lr=c.triplet_lr_init, momentum=c.MOMENTUM,
                        weight_decay=c.WEIGHT_DECAY)
gamma_triplet = 10 ** (np.log10(c.triplet_lr_last / c.triplet_lr_init) / (c.triplet_epoch_num - 1))
lr_scheduler_triplet = optim.lr_scheduler.StepLR(optimizer_triplet, step_size=1, gamma=gamma_triplet)

criterion_pretrain = nn.CrossEntropyLoss()
optimizer_pretrain = optim.SGD(net.parameters(), lr=c.pretrain_lr_init, momentum=c.MOMENTUM,
                           weight_decay=c.WEIGHT_DECAY)
gamma_pretrain = 10 ** (np.log10(c.pretrain_lr_last / c.pretrain_lr_init) / (c.pretrain_epoch_num - 1))
lr_scheduler_pretrain = optim.lr_scheduler.StepLR(optimizer_pretrain, step_size=1, gamma=gamma_pretrain)

# %% train and test
# def checkpoint():
# 	# load checkpoint model
# 	for epoch in range(15, 20, 2):
# 		print('epoch:%d/%d' % (epoch, 30))
# 		pretrained_dict = torch.load('./data/vox1_model_save/VGG/pretrain_256/epoch_%d'%epoch)
# 		net.load_state_dict(pretrained_dict)
# 		evaluation()


def evaluation():
	# 由于网络中使用了全局平均池化层，所以可以输入任意帧数的语谱图(512×N)。
	# 测试时将整段测试语音输入，由于输入语谱图宽度不一致，所以无法输入一个mini-batch的数据，
	# 但是，为了加快速度(我的服务器可以使用6块GPU)，我们可以设置batch_size=6，由于6块GPU每块都有一个完整的
	# 网络(nn.DataParallel)，在每块GPU上分别计算一个可变宽度的语谱图，变相实现了每次迭代输入6个数据，一定程度上
	# 减少了计算时间。为了构造长度为6的input，预先设置max_width，我们需要对宽度不足max_width的语谱图进行补0(按列)，
	# 并在喂进网络之间还原出原始的语谱图，这就需要知道补0矩阵的列数，因此，我们在最后一列再添加一列关于补0列数
	# 的相关信息，所以最终一个batch数据的维度为(6, 1, 512, 1001)，(假设max_width=1000)
	
	# Get embeddings
	with torch.no_grad():
		net.eval()
		
		embeddings = {}
		for i, (keys_list, spec) in tqdm(enumerate(testset_loader)):
			spec = spec.to(c.device)
			embedding = net(spec, 'evaluation')
			for j in range(len(spec)):
				embeddings[keys_list[j]] = embedding[j:(j + 1), :]
		
	# Compute similarity
	result = []
	targets = []
	pred_similarity_cos = []
	cos = torch.nn.CosineSimilarity(dim=1)
	
	test_pairs_path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_Verification.txt')
	test_pairs = pd.read_table(test_pairs_path, sep=' ', header=None)
	
	for i in tqdm(range(len(test_pairs))):
		target, voice1_path, voice2_path = test_pairs.iloc[i]
		embedding1 = embeddings[voice1_path]
		embedding2 = embeddings[voice2_path]
		
		similarity_cos = cos(embedding1, embedding2)
		# dist_l2 = F.pairwise_distance(embedding1, embedding2)
		## distance to similarity
		# similarity_l2 = -1 * dist_l2
		
		targets.append(target)
		# pred_similarity_l2.append(similarity_l2.item())
		pred_similarity_cos.append(similarity_cos.item())
	# EER1, minDCF1 = compute_eer(pred_similarity_l2, targets)
	EER, minDCF = compute_eer(pred_similarity_cos, targets)
	result.append(EER)
	result.append(minDCF)
	
	return result


def pretrain():
	if not os.path.exists(os.path.join(c.save_path, 'pretrain')):
		os.mkdir(os.path.join(c.save_path, 'pretrain'))
	f = open(os.path.join(c.save_path, 'pre_eer.txt'), 'w')
	f.write('epoch \t EER \t min_dcf \n')
	
	for epoch in range(1, c.pretrain_epoch_num + 1):
		# ------------------------------------------------------------------------------------------- #
		# Pretrain
		# ------------------------------------------------------------------------------------------- #
		print('epoch:%d/%d' % (epoch, c.pretrain_epoch_num))
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
		if epoch > 14:
			torch.save(net.state_dict(), c.save_path + '/pretrain/epoch_%d' % (epoch))
			
			result = evaluation()
			line = str(epoch) + '\t' + '%.6f'%result[0] + '\t' + '%.6f'%result[1]
			f.write(line + '\n')
			print('test set:'
			      '\n EER:%.2f  minDCF:%.3f' % (result[0], result[1]))
	f.close()


def train_triplet():
	# TODO:
	pretrained_dict = torch.load('./data/vox1_model_save/pretrain/epoch_19')
	net.load_state_dict(pretrained_dict)

	if not os.path.exists(os.path.join(c.save_path, 'triplet')):
		os.mkdir(os.path.join(c.save_path, 'triplet'))
	f = open(os.path.join(c.save_path, 'tri_eer.txt'), 'w')
	f.write('epoch \t EER \t min_dcf \n')
	
	for epoch in range(1, c.triplet_epoch_num + 1):
		# ------------------------------------------------------------------------------------------------- #
		# Train triplet loss
		# ------------------------------------------------------------------------------------------------- #
		print('epoch:%d/%d' % (epoch, c.triplet_epoch_num))
		losses = []
		num_triplets = 0
		ap_dist_list = []
		an_dist_list = []
		lr_scheduler_triplet.step()
		net.train()  # batchnorm
		
		# triplet selector
		if c.triplet_selector == 'Batch Hard':
			triplet_selector = HardestNegativeTripletSelector(margin=c.margin,
                                    all_positive=True, squared=True)  # use all anchor-positive pair or not
		elif c.triplet_selector == 'BatchAll':
			triplet_selector = BatchAllTripletSelector(margin=c.margin, squared=True)
		elif c.triplet_selector == 'DistanceWeightedSampling':
			triplet_selector = DistanceWeightedSampling(c.n_samples, normalize=True)
		elif c.triplet_selector == 'Semihard':
			triplet_selector = SemihardNegativeTripletSelector(margin=c.margin, squared=True)
		criterion_triplet = OnlineTripletLoss(margin=c.margin,
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
		                                                                                      c.triplet_epoch_num,
		                                                                                      np.mean(losses),
		                                                                                      num_triplets,
		                                                                                      np.mean(ap_dist_list),
		                                                                                      np.mean(an_dist_list)))
		
		# ------------------------------------------------------------------------------------------------ #
		# Evaluation
		# ------------------------------------------------------------------------------------------------ #
		# Full length utterances are used for test
		if epoch > 5:
			torch.save(net.state_dict(), c.save_path + '/triplet/epoch_%d' % (epoch))
			
			result = evaluation()
			line = str(epoch) + '\t' + '%.6f' % result[0] + '\t' + '%.6f' % result[1]
			f.write(line + '\n')
			print('test set:'
			      '\n EER:%.2f  minDCF:%.3f' % (result[0], result[1]))
	f.close()


if __name__ == '__main__':
	spec.save_feature('./data/feature_npy/test/')
	if c.train_mode == 'pretrain':
		pretrain()
	elif c.train_mode == 'triplet':
		train_triplet()