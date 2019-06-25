VoxCeleb1_Dir = '/data/corpus/VoxCeleb'
VoxCeleb2_Dir = '/data/corpus/VoxCeleb2/dev/aac'

# Device configuration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2, 3, 4, 5]

# Signal Processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
SEGMENT_LEN = 3

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
