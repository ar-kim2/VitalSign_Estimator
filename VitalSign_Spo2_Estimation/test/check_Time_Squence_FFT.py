import numpy as np
import matplotlib.pyplot as plt
import os


from scipy.signal import butter, lfilter
from scipy import fftpack

import csv

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
file_name1 = '0127'
file_name2 = 'TPR1_ar'

############ Chcek PPG Signal ##############
f = open(path + '/../check_data/{}/{}/PPG/ppg_wave_540.csv'.format(file_name1, file_name2), encoding='utf-8')

rdr = csv.reader(f)

ppg_wave = []

for line in rdr:

    if line[0] != 'Wave':
        ppg_wave.append(int(line[0]))

f.close()

print(len(ppg_wave))
ppg_wave = ppg_wave[540:]
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
raw_data_cr = np.load(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_cl = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2), allow_pickle=True)

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
fft_time_interval = 500 # 10s

forehead_data = np.copy(roi_data3_f[:, 7])[time_start_idx:time_start_idx+fft_time_interval]
cheekl_data = np.copy(roi_data3_cl[:, 7])[time_start_idx:time_start_idx+fft_time_interval]
cheekr_data = np.copy(roi_data3_cr[:, 7])[time_start_idx:time_start_idx+fft_time_interval]

bp_data = np.copy(roi_data3_f[:, 6])
# bp_data2 = butter_lowpass_filter(bp_data, 3, 50)
bp_data2 = butter_bandpass_filter(bp_data, 0.5, 3, 50)

# The FFT of the signal
sig_fft = fftpack.fft(forehead_data)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(len(forehead_data), d=(1 / fs))

mask = sample_freq > 0

temp_power = np.array(power[mask])
start_idx = int(np.where(1 <= sample_freq[mask])[0][0])
end_idx = int(np.where(sample_freq[mask] <= 2)[0][-1])

max_power = np.round(np.max(temp_power[start_idx:end_idx]),3)
max_arg = np.argmax(temp_power[start_idx:end_idx])
max_freq = np.round(sample_freq[mask][start_idx+max_arg],3)

plt.figure()
plt.title("max freq: {} max vlaue : {}".format(max_freq, max_power))
plt.plot(sample_freq[mask], power[mask])
# plt.axvspan(1, 2, facecolor='red', alpha=0.5)
# plt.ylim([-0.01, 100])
# plt.ylim([-0.01, 1])
plt.xlabel('Frequency [Hz]')
plt.ylabel('power')
plt.show()
