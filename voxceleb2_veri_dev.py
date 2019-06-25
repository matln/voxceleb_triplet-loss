
import os
import constants as c
import numpy as np
import librosa


# Creating split txt file
f = open('./voxceleb2_veri_dev.txt', 'w')

files = np.sort(os.listdir(c.VoxCeleb2_Dir))

label = 0
for file in files:
	for audio_name in np.sort(os.listdir(os.path.join(c.VoxCeleb2_Dir, file))):
		for audio_segment in np.sort(os.listdir(os.path.join(c.VoxCeleb2_Dir, file, audio_name))):
			audio_path = os.path.join(file, audio_name, audio_segment)
			
			f.write(str(label) + ' ' + audio_path + '\n')
	label += 1
f.close()