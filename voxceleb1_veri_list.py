"""
	Get a txt file which is like [name, path, segment/num_segment]
	train dataset include all speaker' audio except that the speaker whose name starts with 'E'
"""

import os
import constants as c
import numpy as np
import librosa


# Creating split txt file
# TODO:
# f = open('./voxceleb1_veri_dev.txt', 'w')
f = open('./voxceleb1_veri_test.txt', 'w')

path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
speakers = np.sort(os.listdir(path))

# TODO:
# # filtering out the speaker whose name starts with 'E'
# speakers = list(filter(lambda x: x[:1] != 'E', speakers))
speakers = list(filter(lambda x: x[:1] == 'E', speakers))

# label
label_dict = {}
f1 = open("./speaker_id.txt", 'r')
try:
	for lines in f1:
		line = lines.split(' ')
		label_dict[line[0]] = int(line[1])
finally:
	f1.close()

for speaker in speakers:
	for spk_utt in np.sort(os.listdir(os.path.join(path, speaker))):
		# label
		label = label_dict[speaker]
		if label > 308:
			label -= 40

		# path
		wav_path = os.path.join(speaker, spk_utt)
		f.write(str(label) + ' ' + wav_path + '\n')
f.close()