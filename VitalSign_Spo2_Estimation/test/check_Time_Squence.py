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
###############################################

file_name = 'TPR1_ar'
raw_data3_f = np.load(path + '/../check_data/0127/{}/re_forehead_human.npy'.format(file_name), allow_pickle=True)
raw_data3_c = np.load(path + '/../check_data/0127/{}/re_cheek_human.npy'.format(file_name), allow_pickle=True)
raw_data3_eye = np.load(path + '/../check_data/0127/{}/re_under_eye_human.npy'.format(file_name), allow_pickle=True)
raw_data3_nose = np.load(path + '/../check_data/0127/{}/re_under_nose_human.npy'.format(file_name), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

print("CHECK shape : ", np.shape(raw_data3_f))

roi_data3_f = []
for i in range(len(raw_data3_f)):
    avg_data = np.average(np.reshape(raw_data3_f[i][:14, :, :], (14, -1)), axis=1)
    roi_data3_f.append(avg_data)
roi_data3_f = np.array(roi_data3_f)

roi_data3_cr = []
for i in range(len(raw_data3_c)):
    roi_data3_cr.append(np.average(np.reshape(raw_data3_c[i][:14, :, :], (14,-1)), axis=1))
roi_data3_cr = np.array(roi_data3_cr)

roi_data3_eye = []
for i in range(len(raw_data3_eye)):
    roi_data3_eye.append(np.average(np.reshape(raw_data3_eye[i][:14, :, :], (14,-1)), axis=1))
roi_data3_eye = np.array(roi_data3_eye)

roi_data3_nose = []
for i in range(len(raw_data3_nose)):
    roi_data3_nose.append(np.average(np.reshape(raw_data3_nose[i][:14, :, :], (14,-1)), axis=1))
roi_data3_nose = np.array(roi_data3_nose)


#######################################

wavelength = [490.83, 501.66, 517.64, 529.52, 554.8, 564.59, 574.99, 586.3, 602.75, 609.2, 668.79, 687.2747, 702.84, 710.18]


ref_data = np.array(roi_data3_f)

# ref_data2 = butter_lowpass_filter(ref_data, 3, 60)
ref_data2 = butter_bandpass_filter(roi_data3_f[:3000, 7], 0.5, 3, 48)


mvf_data = []
for i in range(len(ref_data)):
    if i < 20:
        mvf_data.append(ref_data[i][4])
    else:
        mvf_data.append(np.average(ref_data[i - 20:i][4]))

end_idx = 3000
time_stamp = []
for i in range(end_idx):
    time_stamp.append(i/50)

plt.figure()
plt.title("Visible")
# plt.plot(time_stamp, roi_data3_f[:end_idx, 7], label='forehead')
# plt.plot(time_stamp, roi_data3_eye[:end_idx, 7], label='under eye')
# plt.plot(time_stamp, roi_data3_cr[:end_idx, 7], label='cheek')
# plt.plot(time_stamp, roi_data3_nose[:end_idx, 7], label='under nose')
plt.plot(time_stamp[500:], ref_data2[500:], label='under nose')
plt.legend()
plt.show()


# plt.figure()
# plt.title("NIR")
# plt.plot(time_stamp, roi_data3_f[:, 11], label='forehead')
# plt.plot(time_stamp, roi_data3_cr[:, 11], label='cheek_r')
# plt.plot(time_stamp, roi_data3_cl[:, 11], label='cheek_l')
# plt.legend()
# plt.show()

# np.save(path + '/input_data/TPR_11_forehead.npy', roi_data3_f)
# np.save(path + '/input_data/TPR_11_cheek_r.npy', roi_data3_cr)
# np.save(path + '/input_data/TPR_11_cheek_l.npy', roi_data3_cl)