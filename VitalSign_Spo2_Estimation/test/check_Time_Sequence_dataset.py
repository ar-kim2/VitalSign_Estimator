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
file_name1 = '0128'
file_name2 = 'TPR1_ch'

############ Chcek PPG Signal ##############
f = open(path + '/../check_data/{}/{}/PPG/ppg_wave_420.csv'.format(file_name1, file_name2), encoding='utf-8')

rdr = csv.reader(f)

ppg_wave = []

for line in rdr:

    if line[0] != 'Wave':
        ppg_wave.append(int(line[0]))

f.close()

print(len(ppg_wave))
ppg_wave = ppg_wave[660:]
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

fs = 1/np.average(sub_time)
print("Interval Time : ", np.average(sub_time), " FPS : ", 1/np.average(sub_time))
############################


############ Read Pulse ##############
f = open(path + '/../check_data/{}/{}/PPG/pulse_sto.csv'.format(file_name1, file_name2), encoding='utf-8')
rdr = csv.reader(f)

pulse_list = []

first_time = 0
for line in rdr:
    if line[3] != 'PULSE':
        str_time = line[3]
        pulse = int(str_time)
        pulse_list.append(pulse)
f.close()

pulse_list = np.array(pulse_list)


################ Measurement data ########################################
raw_data_f = np.load(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_c = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_eye = np.load(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data_nose = np.load(path + '/../check_data/{}/{}/re_under_nose_human.npy'.format(file_name1, file_name2), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

roi_data3_f = []
for i in range(len(raw_data_f)):
    avg_data = np.average(np.reshape(raw_data_f[i][:14, :, :], (14, -1)), axis=1)
    roi_data3_f.append(avg_data)
roi_data3_f = np.array(roi_data3_f)

roi_data3_c = []
for i in range(len(raw_data_c)):
    roi_data3_c.append(np.average(np.reshape(raw_data_c[i][:14, :, :], (14,-1)), axis=1))
roi_data3_c = np.array(roi_data3_c)

roi_data3_eye = []
for i in range(len(raw_data_eye)):
    roi_data3_eye.append(np.average(np.reshape(raw_data_eye[i][:14, :, :], (14,-1)), axis=1))
roi_data3_eye = np.array(roi_data3_eye)

roi_data3_nose = []
for i in range(len(raw_data_nose)):
    roi_data3_nose.append(np.average(np.reshape(raw_data_nose[i][:14, :, :], (14,-1)), axis=1))
roi_data3_nose = np.array(roi_data3_nose)

#######################################

re_pulse_list = []
for i in range(len(pulse_list)):
    if pulse_list[i] < 200:
        re_pulse_list.append(pulse_list[i])

print("Pulse Average Variance Min Max")
print("{}\t{}\t{}\t{}".format(np.average(re_pulse_list), np.std(re_pulse_list), np.min(re_pulse_list), np.max(re_pulse_list)))

#########################################


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

plt.figure()
plt.subplot(5, 1, 1)
# plt.plot(measure_time[:len(roi_data3_f)], roi_data3_f[:, 6], label='forehead')
plt.plot(roi_data3_f[:, 6], label='forehead')
# plt.plot(measure_time[:len(roi_data3_f)], roi_data3_cr[:, 6], label='cheek_r')
# plt.plot(measure_time[:len(roi_data3_cl)], roi_data3_cl[:, 6], label='cheek_l')
# plt.plot(measure_time[:len(roi_data3_cl)], roi_data3_cl[:, 6], label='cheek_l')
# plt.plot(measure_time[500:len(bp_data2)], bp_data2[500:], label='cheek_l')
plt.legend()

plt.subplot(5, 1, 2)
# plt.plot(measure_time[:len(roi_data3_f)], roi_data3_eye[:, 6], label='forehead')
plt.plot(roi_data3_eye[:, 6], label='under eye')
plt.legend()

plt.subplot(5, 1, 3)
# plt.plot(measure_time[:len(roi_data3_f)], roi_data3_c[:, 6], label='forehead')
plt.plot(roi_data3_c[:, 6], label='cheek')
plt.legend()

plt.subplot(5, 1, 4)
# plt.plot(measure_time[:len(roi_data3_f)], roi_data3_nose[:, 6], label='forehead')
plt.plot(roi_data3_nose[:, 6], label='under nose')
plt.legend()

plt.subplot(5, 1, 5)
# plt.plot(ppg_time[:ppg_end_time], ppg_wave[:ppg_end_time])
plt.plot(pulse_list[:ppg_end_time])
plt.ylim([50, 120])
# plt.plot(ppg_time[:len(roi_data3_f)], ppg_wave[:len(roi_data3_f)])
plt.show()
plt.show()



# plt.figure()
# plt.plot(ppg_time[500:ppg_end_time], (ppg_wave[500:ppg_end_time]-np.average(ppg_wave[500:ppg_end_time]))/10000, label='ppg')
# plt.plot(measure_time[500:len(bp_data2)], bp_data2[500:], label='measure')
#
# plt.legend()
# plt.show()
