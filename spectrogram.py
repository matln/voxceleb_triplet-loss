# Reference:
#       https://github.com/linhdvu14/vggvox-speaker-identification
#       https://github.com/jameslyons/python_speech_features
import librosa
import numpy as np
from scipy.signal import lfilter, butter

import decimal
import math
import constants as c
from tqdm import tqdm
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


def get_spectrum(filename):
	signal = load_wav(filename, c.SAMPLE_RATE)
	signal *= 2 ** 15
	
	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = framesig(signal, frame_len=c.FRAME_LEN * c.SAMPLE_RATE,
	                          frame_step=c.FRAME_STEP * c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.rfft(frames, n=c.NUM_FFT))
	fft = fft.T
	fft = fft.astype(np.float32)
	return fft


def save_feature(save_path=None):
	# No normalize in get_spectrum (do it in transform procedure)
	
	if len(os.listdir(save_path)) == 0:
		print('Save testset spectrogram as *.npy ...')
		path = os.path.join(c.VoxCeleb1_Dir, 'voxceleb1_wav')
		speakers = np.sort(os.listdir(path))
		speakers = list(filter(lambda x: x[:1] == 'E', speakers))
		for speaker in tqdm(speakers):
			os.mkdir(save_path + speaker)
			speaker_utts = os.listdir(os.path.join(path, speaker))
			for i, speaker_utt in enumerate(np.sort(speaker_utts)):
				# Read wav file and extract feature
				wav_path = os.path.join(path, speaker, speaker_utt)
				spec = get_spectrum(wav_path)
				feature_path = os.path.join(save_path, speaker, speaker_utt)
				np.save(feature_path.replace('.wav', '.npy'), spec)
		print('\nFinished...')


if __name__ == '__main__':
	# TODO:
	save_feature('./data/feature_npy/test/')





	
	



	