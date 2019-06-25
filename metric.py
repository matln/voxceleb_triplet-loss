"""
	Define metric function (minDCF, EER)
"""
import numpy as np

def compute_eer(scores, labels, eps=1e-6, showfig=False):
	"""
	If score > threshold, prediction is positive, else negative

	:param scores: similarity for target and non-target trials
	:param labels: true labels for target and non-target trials
	:param showfig: if true, the DET curve is displayed
	:return:
		eer: percent equal eeror rate (EER)
		dcf: minimum detection cost function (DCF) with voxceleb parameters

	:Reference: Microsoft MSRTookit: compute_eer.m
		数据挖掘导论 p183
	:Author: li-jianchen(matln)
	"""
	
	# Get the index list after sorting the scores list
	sorted_index = [index for index, value in sorted(enumerate(scores), key=lambda x: x[1])]
	# Sort the labels list
	sorted_labels = [labels[i] for i in sorted_index]
	sorted_labels = np.array(sorted_labels)
	
	FN = np.cumsum(sorted_labels == 1) / (sum(sorted_labels == 1) + eps)
	TN = np.cumsum(sorted_labels == 0) / (sum(sorted_labels == 0) + eps)
	FP = 1 - TN
	TP = 1 - FN
	
	FNR = FN / (TP + FN + eps)
	FPR = FP / (TN + FP + eps)
	difs = FNR - FPR
	idx1 = np.where(difs < 0, difs, float('-inf')).argmax(axis=0)
	idx2 = np.where(difs >= 0, difs, float('inf')).argmin(axis=0)
	# the x-axis of two points
	x = [FPR[idx1], FPR[idx2]]
	# the y-axis of two points
	y = [FNR[idx1], FNR[idx2]]
	# compute the intersection of the straight line connecting (x1, y1), (x2, y2)
	# and y = x.
	# Derivation: (x-x1) / (x2-x1) = (x-y1) / (y2-y1)                 ->
	#             (x-x1)(y2-y1) = (x-y1)(x2-x1)                       ->
	#             x(y2-y1-x2-x1) = x1(y2-y1) - y1(x2-x1)              ->
	#                            = x1(x2-x1) - y1(x2-x1)
	#                              + x1(y2-y1) - x1(x2-x1)            ->
	#                            = (x1-x2)(x2-x1) + x1(y2-y1-x2+x1)   ->
	#             x = x1 + (x1-x2)(x2-x1) / (y2-y1-x2-x1)
	a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
	eer = 100 * (x[0] + a * (y[0] - x[0]))
	
	## Compute dcf
	# VoxCeleb performance parameter
	Cmiss = 1
	Cfa = 1
	P_tgt = 0.01
	
	Cdet = Cmiss * FNR * P_tgt + Cfa * FPR * (1 - P_tgt)
	dcf_voxceleb = 100 * min(Cdet)
	
	## figure
	if showfig:
		plot_det(FPR, FNR)
	
	return eer, dcf_voxceleb


def plot_det(FPR, FNR):
	# Plots the detection error tradeoff (DET) curve
	# Reference: compute_eer.m
	pass