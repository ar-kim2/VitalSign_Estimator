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
file_name1 = 'total_data'
file_name2 = '0204_TPR5_ar_2'

############ Chcek PPG Signal ##############
# f = open(path + '/../check_data/{}/{}/PPG/ppg_wave_480.csv'.format(file_name1, file_name2), encoding='utf-8')
f = open(path + '/../check_data/{}/{}/ppg_wave.csv'.format(file_name1, file_name2), encoding='utf-8')

rdr = csv.reader(f)

ppg_wave = []

for line in rdr:

    if line[0] != 'Wave':
        ppg_wave.append(int(line[0]))

f.close()

# ppg_wave = ppg_wave[480:]
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
        hour = int(str_time[-15:-13])
        min = int(str_time[-12:-10])
        sec = int(str_time[-9:-7])
        ms = int(str_time[-6:-1])

        total_time = hour*60*60*100000+min*60*100000+sec*100000+ms

        if line[0] == '0':
            first_time = total_time

        measure_time.append((total_time-first_time)/100000)

        if measure_time[-1] < 0:
            print("min {} sec {} ms {} idx {}".format(min, sec, ms, line))

f.close()

###### Calculate Fps ########
sub_time = []
for i in range(1, 100):
    sub_time.append(measure_time[i]-measure_time[i-1])

print("Interval Time : ", np.average(sub_time), " FPS : ", 1/np.average(sub_time))
############################

################ Measurement data ########################################
raw_data_f = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_cr = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_cl = np.load(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name2), allow_pickle=True)

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

# 574.99
wavelength = [490.83, 501.66, 517.64, 529.52, 554.8, 564.59, 574.99, 586.3, 602.75, 609.2, 668.79, 687.2747, 702.84, 710.18]

bp_data = np.copy(roi_data3_f[:, 6])
# bp_data2 = butter_lowpass_filter(bp_data, 3, 50)
bp_data2 = butter_bandpass_filter(bp_data, 0.5, 3, 50)

# bf_data = butter_lowpass_filter(roi_data, 3, fs)
# # bf_data = butter_bandpass_filter(roi_data, 0.5, 3, fs)

# print(np.where(np.array(ppg_time) > measure_time[-1])[0][0])
# print(np.where(np.array(ppg_time) > measure_time[-1])[0][-1])
ppg_end_time = int(np.where(np.array(ppg_time) > measure_time[len(roi_data3_f)])[0][0])

# plt.figure()
# plt.subplot(2, 1, 1)
# # plt.plot(measure_time[:len(bp_data2)], bp_data2, label='cheek_r')
# # plt.plot(measure_time[:len(roi_data3_f)], roi_data3_f[:, 6], label='cheek')
# # plt.plot(measure_time[:len(roi_data3_f)], roi_data3_cr[:, 6], label='cheek_r')
# # plt.plot(measure_time[:len(roi_data3_cl)], roi_data3_cl[:, 6], label='cheek_l')
# # plt.plot(measure_time[:len(roi_data3_cl)], roi_data3_cl[:, 6], label='cheek_l')
# plt.plot(measure_time[500:len(bp_data2)], bp_data2[500:], label='cheek_l')
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(ppg_time[500:ppg_end_time], ppg_wave[500:ppg_end_time])
# # plt.plot(ppg_time[:len(roi_data3_f)], ppg_wave[:len(roi_data3_f)])
# plt.show()


plt.figure()
plt.plot(ppg_time[:ppg_end_time], ppg_wave[:ppg_end_time], label='ppg')
# plt.plot(measure_time[500:len(bp_data2)], bp_data2[500:], label='measure')
plt.legend()
plt.show()

print("PPG Min Max")
print("{}\t{}\t".format(np.min(ppg_wave[:ppg_end_time]), np.max(ppg_wave[:ppg_end_time])))
