import numpy as np
import matplotlib.pyplot as plt
import os


from scipy.signal import butter, lfilter
from scipy import fftpack

import csv


path = os.path.dirname(__file__)
file_name = '0204_TPR5_ar_2'


############ Chcek PPG Signal ##############
f = open(path + '/../check_data/total_data/{}/ppg_wave.csv'.format(file_name), encoding='utf-8')

rdr = csv.reader(f)

ppg_wave = []

for line in rdr:

    if line[0] != 'Wave':
        ppg_wave.append(int(line[0]))

f.close()

ppg_time = []
for i in range(len(ppg_wave)):
    ppg_time.append(i*(1/60))


# Draw PPG Signal
plt.figure()
plt.title('PPG Signal')
plt.plot(ppg_time, ppg_wave)
plt.xlabel('time (s)')
plt.ylabel('PPG')
plt.show()

###############################################

############ Read Pulse ##############
f = open(path + '/../check_data/total_data/{}/pulse_sto.csv'.format(file_name), encoding='utf-8')
rdr = csv.reader(f)

pulse_list = []
Spo2_list = []

first_time = 0
for line in rdr:
    if line[3] != 'PULSE':
        str_time = line[3]
        pulse = int(str_time)
        spo2 = int(line[2])
        pulse_list.append(pulse)
        Spo2_list.append(spo2)
f.close()

pulse_list = np.array(pulse_list)
Spo2_list = np.array(Spo2_list)

print("Pulse Average Variance Min Max")
print("{}\t{}\t{}\t{}".format(np.average(pulse_list), np.std(pulse_list), np.min(pulse_list), np.max(pulse_list)))

print("Spo2 Average Variance Min Max")
print("{}\t{}\t{}\t{}".format(np.average(Spo2_list[:-10]), np.std(Spo2_list[:-10]), np.min(Spo2_list[:-10]), np.max(Spo2_list[:-10])))

# Draw Pulse graph
plt.figure()
plt.title('Pulse')
plt.plot(pulse_list)
plt.xlabel('time (s)')
plt.ylabel('pulse')
plt.show()

# Draw Pulse graph
plt.figure()
plt.title('Spo2')
plt.plot(Spo2_list[:-10])
plt.xlabel('time (s)')
plt.ylabel('Spo2')
plt.show()


############################

################ Measurement data ########################################
# Raw Data Shape [time][wavelength][roi width][roi height]

raw_data_f = np.load(path + '/../check_data/total_data/{}/re_forehead_human.npy'.format(file_name), allow_pickle=True)
raw_data_c = np.load(path + '/../check_data/total_data/{}/re_cheek_human.npy'.format(file_name), allow_pickle=True)
raw_data_eye = np.load(path + '/../check_data/total_data/{}/re_under_eye_human.npy'.format(file_name), allow_pickle=True)
raw_data_nose = np.load(path + '/../check_data/total_data/{}/re_under_nose_human.npy'.format(file_name), allow_pickle=True)

############ ROI Detection ############
# roi_data Shape [time][wavelength]

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

############ Measurement Elapsed Time ##############
f = open(path + '/../check_data/total_data/{}/time_stamp.csv'.format(file_name), encoding='utf-8')
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


# Draw Measurment graph
wavelength_index = 6

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(measure_time[:len(roi_data3_f)], roi_data3_f[:, 6], label='forehead')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(measure_time[:len(roi_data3_eye)], roi_data3_eye[:, 6], label='under eye')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(measure_time[:len(roi_data3_c)], roi_data3_c[:, 6], label='cheek')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(measure_time[:len(roi_data3_nose)], roi_data3_nose[:, 6], label='under nose')
plt.legend()
plt.show()

