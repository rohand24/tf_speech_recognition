from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal

(rate, sig) = wav.read("D:\\Kaggle_Speech_Recognition\\Datasets\\train\\audio\\bed\\00f0204f_nohash_0.wav")
mfcc_feat = mfcc(sig, rate)
print(mfcc_feat[1:3,:])
samples_rate, samples = wav.read("D:\\Kaggle_Speech_Recognition\\Datasets\\train\\audio\\bed\\00f0204f_nohash_1.wav")
freq, time, spectrogram = signal.spectrogram(samples, samples_rate)
plt.pcolormesh(time, freq, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()