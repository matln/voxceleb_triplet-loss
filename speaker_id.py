import os
import constants as c

f = open("./speaker_id.txt", 'w')

path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
label_list = []
files = os.listdir(path)
files.sort()
id = 0
for file in files:
    label_list.append([file, id])
    id += 1

for i in range(len(label_list)):
    f.write(label_list[i][0] + ' ' + str(label_list[i][1]) + '\n')
f.close()