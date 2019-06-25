"""
	Get a txt file which is like [name, path, segment/num_segment]
	train dataset include all speaker' audio except that the speaker whose name starts with 'E'
"""

import os
import constants as c
import numpy as np
import librosa


# Creating split txt file
f = open('./voxceleb1_verification_trainset.txt', 'w')

path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
files = np.sort(os.listdir(path))
# filtering out the speaker whose name starts with 'E'
files = list(filter(lambda x: x[:1] != 'E', files))

# label
label_dict = {}
f1 = open("./speaker_id.txt", 'r')
try:
	for lines in f1:
		line = lines.split(' ')
		label_dict[line[0]] = int(line[1])
finally:
	f1.close()

for file in files:
	for file_feature in np.sort(os.listdir(os.path.join(path, file))):
		# feature_path = os.path.join(path, file, file_feature)
		# img = np.load(feature_path)
		# num_segment = int(len(img) / c.NUM_FRAMES)

		# new identification split txt format:
		#   label   path    segment/num_segment

		# label
		name = file
		label = label_dict[name]
		if label > 308:
			label -= 40

		# path
		# segment_path = os.path.join(file, file_feature.replace('.npy', '.wav'))
		# for segment in range(num_segment):
		# 	f.write(str(label) + ' ' + segment_path + ' ' + str(segment+1) + '/' + str(num_segment) + '\n')
		
		wav_path = os.path.join(file, file_feature)
		f.write(str(label) + ' ' + wav_path + '\n')
		
	
		
		# wav_path = os.path.join(file, file_wav)
		# # label
		# name = file
		# label = label_dict[name]
		# if label > 308:
		# 	label -= 40
		# f.write(str(label) + ' ' + wav_path + '\n')
f.close()