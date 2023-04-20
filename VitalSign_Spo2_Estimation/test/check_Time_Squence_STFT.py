import numpy as np
import matplotlib.pyplot as plt
import os


from scipy.signal import butter, lfilter
from scipy import fftpack

import csv

from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=9):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=9):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



path = os.path.dirname(__file__)
file_name1 = '0118_Surface_Ref'
file_name2 = 'TPR3_full'

############ Chcek PPG Signal ##############
f = open(path + '/../check_data/{}/{}/PPG/ppg_wave_3220.csv'.format(file_name1, file_name2), encoding='utf-8')

rdr = csv.reader(f)

ppg_wave = []

for line in rdr:

    if line[0] != 'Wave':
        ppg_wave.append(int(line[0]))

f.close()

print(len(ppg_wave))
ppg_wave = ppg_wave[3220:]
ppg_time = []
for i in range(len(ppg_wave)):
    ppg_time.append(i*(1/60))


###############################################


############ Measurement Elapsed Time ##############
f = open(path + '/../check_data/{}/{}/time_stamp.csv'.format(file_name1, file_name2), encoding='utf-8')
rdr = csv.reader(f)

measure_time = []

first_time = 0
for line in rdr:
    if line[0] != 'number':
        str_time = line[1]
        min = int(str_time[-12:-10])
        sec = int(str_time[-9:-7])
        ms = int(str_time[-6:-1])

        total_time = min*60*100000+sec*100000+ms

        if line[0] == '0':
            first_time = total_time

        measure_time.append((total_time-first_time)/100000)

f.close()

###### Calculate Fps ########
sub_time = []
for i in range(1, 100):
    sub_time.append(measure_time[i]-measure_time[i-1])

print("Interval Time : ", np.average(sub_time), " FPS : ", 1/np.average(sub_time))

fs = 1/np.average(sub_time)
############################


################ Measurement data ########################################
raw_data_f = np.load(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_cr = np.load(path + '/../check_data/{}/{}/re_cheek_r_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_cl = np.load(path + '/../check_data/{}/{}/re_cheek_l_human.npy'.format(file_name1, file_name2), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

roi_data3_f = []
for i in range(len(raw_data_f)):
    avg_data = np.average(np.reshape(raw_data_f[i][:14, :, :], (14, -1)), axis=1)
    roi_data3_f.append(avg_data)
roi_data3_f = np.array(roi_data3_f)

roi_data3_cr = []
for i in range(len(raw_data_cr)):
    roi_data3_cr.append(np.average(np.reshape(raw_data_cr[i][:14, :, :], (14,-1)), axis=1))
roi_data3_cr = np.array(roi_data3_cr)

roi_data3_cl = []
for i in range(len(raw_data_cl)):
    roi_data3_cl.append(np.average(np.reshape(raw_data_cl[i][:14, :, :], (14,-1)), axis=1))
roi_data3_cl = np.array(roi_data3_cl)

#######################################

time_start_idx = 0
fft_time_interval = 3000 # 10s

forehead_data = np.copy(roi_data3_f[:, 6])[time_start_idx:time_start_idx+fft_time_interval]
cheekl_data = np.copy(roi_data3_cl[:, 6])[time_start_idx:time_start_idx+fft_time_interval]
cheekr_data = np.copy(roi_data3_cr[:, 6])[time_start_idx:time_start_idx+fft_time_interval]

bp_data = butter_bandpass_filter(roi_data3_f[:, 6], 0.5, 3, fs)
bp_data2 = np.array(bp_data)[time_start_idx+500:time_start_idx+fft_time_interval+500]
# print(np.shape(bp_data))

plt.figure()
plt.plot(bp_data)
plt.show()

print("CHECK fore head data shape : ", np.shape(forehead_data))
print("CHECK bandpass data shape : ", np.shape(bp_data2))

# f, t, Sxx = signal.spectrogram(forehead_data, fs, nperseg = 180 , nfft= 420, noverlap= 179)
f, t, Sxx = signal.spectrogram(bp_data2, fs, nperseg = 180 , nfft= 420, noverlap= 179)


plt.figure()
plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.pcolormesh(t, f, Sxx, shading='gouraud', vmax=1e-4)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


