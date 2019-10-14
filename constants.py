VoxCeleb1_Dir = '/data/corpus/VoxCeleb'
# VoxCeleb2_Dir = '/data/corpus/VoxCeleb2/dev/aac'

save_path = './data/vox1_model_save'
triplet_selector = 'Semihard'         # 'Batch Hard', 'BatchAll', 'Semihard', 'DistanceWeightedSampling'
train_mode = 'triplet'         # 'pretrain', 'triplet'

# Device configuration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2, 3, 4]

embedding_dim = 256    # or 128
batch_size = 64
num_workers = 16
# If GPU memory is not enough, you can decrease `n_classes` or `n_samples`
n_classes = 30
n_samples = 10
pretrain_lr_init = 0.01
pretrain_lr_last = 0.0001
pretrain_epoch_num = 30
margin = 0.3
triplet_lr_init = 0.005
triplet_lr_last = 0.00005
triplet_epoch_num = 30



# Signal Processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
SEGMENT_LEN = 3

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
