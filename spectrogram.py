# Reference:
#       https://github.com/linhdvu14/vggvox-speaker-identification
#       https://github.com/jameslyons/python_speech_features
import librosa
import numpy as np
from scipy.signal import lfilter, butter
import pandas as pd
import random
import decimal
import math
import constants as c
import torch
import os


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        return []
    else:
        numframes = 1 + int(math.floor((1.0 * slen - frame_len) / frame_step)) # LV     # math.ceil

    # padlen = int((numframes - 1) * frame_step + frame_len)
	#
    # zeros = np.zeros((padlen - slen,))
    # padsignal = np.concatenate((sig, zeros))
    padsignal = sig
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1, -1], [1, -alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


class VAD(object):
	# Double threshold method based on short-time zero-crossing rate and short-time energy
	
	def __init__(self, FrameLen=400, FrameInc=160, amp1=10, amp2=2, zcr1=10,
	             zcr2=5, minsilence=40, minlen=15):
		self.FrameLen = FrameLen  # 帧长
		self.FrameInc = FrameInc  # 帧yi
		self.amp1 = amp1  # 短时能量阈值
		self.amp2 = amp2
		self.zcr1 = zcr1  # 过零率阈值
		self.zcr2 = zcr2
		self.minsilence = minsilence  # 用无声的长度来判断语音是否结束
		self.minlen = minlen  # 判断是语音的最小长度
	
	def __call__(self, X):
		x = X / max(abs(X))  # 幅度归一化到[-1, 1]
		
		amp1 = self.amp1
		amp2 = self.amp2
		seg = []
		fragment = []
		while len(x) > 0:  # 剩余采样点大于0时，继续端点检测
			status = 0  # 初始化语音段的状态
			count = 0  # 初始化语音序列的长度
			silence = 0  # 初始化无声的长度
			
			# 计算过零率
			tmp1 = framesig(x[:-1], frame_len=self.FrameLen, frame_step=self.FrameInc)
			tmp2 = framesig(x[1:], frame_len=self.FrameLen, frame_step=self.FrameInc)
			
			if len(tmp1) > 0 and len(tmp2) > 0:
				signs = (tmp1 * tmp2) < 0
				diffs = (tmp1 - tmp2) > 0.02
				zcr = np.sum(signs * diffs, 1)
			

				# 计算短时能量
				amp = np.sum(np.abs(framesig(x, frame_len=400, frame_step=160)) ** 2, 1)
			
				# 调整能量门限
				amp1 = min(amp1, max(amp) / 4)
				amp2 = min(amp2, max(amp) / 8)
			else:
				zcr = []

			# 开始端点检测
			x1 = 0
			for n in range(len(zcr)):
				if status in [0, 1]: # status: 语音段的状态   0 = 静音, 1 = 可能开始
					if amp[n] > amp1: # 确信进入语音段  # 能量超过高阈值)
						x1 = max(n - count, 1)  # 记录语音段的起始点
						status = 2  # 语音段的状态改为2
						silence = 0
						count = count + 1
					elif amp[n] > amp2 or zcr[n] > self.zcr2:     # 可能处于语音段
						status = 1
						count = count + 1
					else:    # 静音状态
						status = 0
				if status == 2:  # 2: 语音段
					if amp[n] > amp2 or zcr[n] > self.zcr2:  # 保持在语音段
						count = count + 1
					else:   # 语音将结束
						silence = silence + 1
						if silence < self.minsilence:    # 静音还不够长，尚未结束
							count = count + 1
						elif count < self.minlen:    # 语音长度太短，认为是噪声
							status = 0
							silence = 0
							count = 0
						else:   # 语音结束
							status = 3
				if status == 3:
					continue
			# print('x1:%d'% x1)
			# print('count:%d' % count)
			count = count - silence / 2

			if x1 == 0:      # 后段全为静音
				x = []
				X = []
			else:
				x2 = x1 + count - 1
				fragment = X[(x1 - 1) * self.FrameInc : int(x2 * self.FrameInc)]
				seg = np.concatenate((seg, fragment))
				x = x[int(x2 * self.FrameInc):]
				X = X[int(x2 * self.FrameInc):]

			finalSeg = seg
		return finalSeg



def get_spectrum(filename):
	# If you need to do VAD(voice activity detection) preprocessing, add the following code:
	#       vad = VAD()
	#       signal = vad(signal)
	# But in out experiments is not used, VoxCeleb1 paper also mentioned that no VAD preprocessing
	# is used.
	
	signal = load_wav(filename, c.SAMPLE_RATE)
	signal *= 2 ** 15
	
	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = framesig(signal, frame_len=c.FRAME_LEN * c.SAMPLE_RATE,
	                          frame_step=c.FRAME_STEP * c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames, n=c.NUM_FFT))
	fft = fft.T
	return fft


def save_feature_vad():
	# Because some utterances are less than 3s after VAD processing, so we concatenate the
	# feature array which has the same Youtube ID and then divide them into 3s segments.
	
	# Need the following code in function get_spectrum():
	#	 vad = VAD()
	#	 signal = vad(signal)
	
	path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
	files = np.sort(os.listdir(path))
	# Only save development dataset
	files = list(filter(lambda x: x[:1] != 'E', files))
	counter = 0
	for file in files:
		counter += 1
		print(counter)
		os.mkdir('./data/spectrogram_vad/' + file)
		file_list = os.listdir(os.path.join(path, file))
		for i, file_wav in enumerate(np.sort(file_list)):
			# Read wav file and extract feature
			wav_path = os.path.join(path, file, file_wav)
			spec_feature, duration1, duration2 = get_spectrum(wav_path)
			if duration2 < 3:
				print('----------------------------------')
				print('path: ', wav_path)
				print('duration1:%f,    duration2:%f' % (duration1, duration2))
			# assert duration2 > 3, 'duration1:%f,    duration2:%f'%(duration1, duration2)
			spec_feature = spec_feature.astype(np.float32)
			
			if i == 0:
				YoutubeID = file_wav.split('_')[0]
				spec_features = spec_feature
			
			# Concatenate same YoutubeID feature array
			if file_wav.split('_')[0] == YoutubeID:
				spec_features = np.concatenate((spec_features, spec_feature), 1)
			
			# New wav file of different YoutubeID
			if file_wav.split('_')[0] != YoutubeID or i == len(file_list) - 1:
				# Save prior spec_features (divided into 3s segment)
				num_segment = math.floor(
					spec_features.shape[1] / 300 + 0.66)  # if the length of the last remain < 100, cut off
				for idx in range(num_segment):
					if idx != math.ceil(spec_features.shape[1] / 300) - 1:
						segment_feature = spec_features[:, idx * 300:(idx + 1) * 300]
					else:
						segment_feature = spec_features[:, spec_features.shape[1] - 300:]
					feature_name = YoutubeID + '_' + str(idx + 1) + '.npy'
					save_path = os.path.join('./data/spectrogram_vad', file, feature_name)
					np.save(save_path, segment_feature)
				
				# Update YoutubeID; if i == len(file_list) - 1, the following code is mearningless
				YoutubeID = file_wav.split('_')[0]
				spec_features = spec_feature


def save_feature(save_path=None):
	# No normalize in get_spectrum (do it in transform procedure)
	
	path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
	files = np.sort(os.listdir(path))
	files = list(filter(lambda x: x[:1] != 'E', files))
	counter = 0
	for file in files:
		counter += 1
		print(counter)
		os.mkdir(save_path + file)
		file_list = os.listdir(os.path.join(path, file))
		for i, file_wav in enumerate(np.sort(file_list)):
			# Read wav file and extract feature
			wav_path = os.path.join(path, file, file_wav)
			spec_feature= get_spectrum(wav_path)
			spec_feature = spec_feature.astype(np.float32)
			feature_path = os.path.join(save_path, file, file_wav)
			np.save(feature_path.replace('.wav', '.npy'), spec_feature)
			
def save_feature_vox2(save_path=None):
	files = np.sort(os.listdir(c.VoxCeleb2_Dir))
	counter = 0
	for file in files:
		counter += 1
		print(counter)
		os.mkdir(save_path + '/' +  file)
		for audio_name in np.sort(os.listdir(os.path.join(c.VoxCeleb2_Dir, file))):
			for audio_segment in np.sort(os.listdir(os.path.join(c.VoxCeleb2_Dir, file, audio_name))):
				audio_path = os.path.join(c.VoxCeleb2_Dir, file, audio_name, audio_segment)
				spec_feature = get_spectrum(audio_path)
				spec_feature = spec_feature.astype(np.float32)
				save_fea_path = os.path.join(save_path, file, audio_segment)
				np.save(save_fea_path.replace('.m4a', '.npy'), spec_feature)



if __name__ == '__main__':
	save_feature('../data/vox1_spectrogram/dev/')
	# save_feature_vox2('../data/vox2_spectrogram/dev')

	# filename = '/data/corpus/VoxCeleb/voxceleb1_wav/A.J._Buckley/1zcIwhmdeo4_0000001.wav'
	# fft = get_spectrum(filename)
	# print(fft[:, 2])
	# print(abs(fft[0,:]))
	
	
	# array = np.load('./data/spectrogram//A.J._Buckley/1zcIwhmdeo4_0000001.npy')
	# print(array[:, 0])
	
	





	# max_ = 0
	# path = './data/spectrogram/test'
	# files = np.sort(os.listdir(path))
	# counter = 0
	# for file in files:
	# 	counter += 1
	# 	print(counter)
	# 	file_list = os.listdir(os.path.join(path, file))
	# 	for i, file_wav in enumerate(np.sort(file_list)):
	# 		# Read wav file and extract feature
	# 		wav_path = os.path.join(path, file, file_wav)
	# 		array = np.load(wav_path)
	# 		if array.shape[1] > max_:
	# 			max_ = array.shape[1]
	# print(max_)
	




	
	



	