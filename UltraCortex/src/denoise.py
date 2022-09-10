import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch, freqz, filtfilt

plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size': 18})


'''
Band pass filters with Butterworth filter
Order = 5
Sampling frequency(fs) = 250 Hz
Nyquist frequency(low, high) = 0.5 * fs
'''
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_folder = os.path.join(project_path, 'UltraCortex', 'data', 'frq_band.csv')

'''
Band pass filters with Butterworth filter
Order = 5
Sampling frequency(fs) = 250 Hz
Nyquist frequency(low, high) = 0.5 * fs
'''
df = pd.read_csv(data_folder)
band_pass_eeg = butter_bandpass_filter(data = df.Fp2, lowcut = 1, highcut = 50, fs = 250, order = 3)


'''
Notch filter to remove in-line noise at ~60 Hz (U.S.A)
Process it twice to remove the 60 Hz spike completely
'''
b_notch, a_notch = iirnotch(59.9250936329588, 20, 250)
filtered_eeg = filtfilt(b_notch, a_notch, band_pass_eeg)

b_notch, a_notch = iirnotch(59.9250936329588, 20, 250)
filtered_eeg = filtfilt(b_notch, a_notch, filtered_eeg)


'''
Compute Fourier Coeffecients: Complex calues of Sin and Cosine 
magintude and phase information

Compute Power Spectrum Density: PSD removes noise floor from signal
'''
number_of_points = len(filtered_eeg)
fhat = np.fft.fft(filtered_eeg, number_of_points)
PSD = fhat * np.conj(fhat) / number_of_points
freq = (1/(0.004*number_of_points))*np.arange(number_of_points)
L = np.arange(1,np.floor(number_of_points/2),dtype='int')

'''
Filter frequency based on PSD (>100 threshold)
Inverse Fourier Transform
'''
## Use the PSD to filter out noise
indices = PSD > 100
PSDclean = PSD * indices  # Zero out all others
fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
ffilt = np.fft.ifft(fhat)

'''
Plotting Denoised data
'''
fig,axs = plt.subplots(3,1)
data_points = np.arange(number_of_points)

plt.sca(axs[0])
plt.plot(data_points,band_pass_eeg,color='k',linewidth=1.5,label='Bandpass filter')
plt.xlim(data_points[0],data_points[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(data_points,ffilt,color='b',linewidth=2,label='FFT Filtered')
plt.xlim(data_points[0],data_points[-1])
plt.legend()

plt.sca(axs[2])
plt.plot(freq[L][3:75],PSD[L][3:75],color='r',linewidth=2,label='Bandpass')
plt.plot(freq[L][3:75],PSDclean[L][3:75],color='b',linewidth=1.5,label='FFT')
plt.legend()
plt.xticks(freq[L][3:75:5])
plt.show()