import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)


# raw_data = np.load(path + '/../check_data/total_data1/TPR2_js/re_forehead_human.npy', allow_pickle=True)
raw_data = np.load(path + '/../check_data/total_data1_2/TPR9_sj_2/re_forehead_human.npy', allow_pickle=True)




plt.figure()
plt.plot(np.mean(raw_data[:, 7, 10:20, 10:20], axis=(1, 2)))
plt.show()

print("CHECK ", np.shape(raw_data))

exit()


###############################################
raw_data_f = np.load(path + '/../check_data/0118_Surface_Ref/TPR4_3/re_forehead_human.npy', allow_pickle=True)
raw_data_cl = np.load(path + '/../check_data/0118_Surface_Ref/TPR4_2/re_cheek_l_human.npy', allow_pickle=True)
raw_data_cr = np.load(path + '/../check_data/0118_Surface_Ref/TPR4_2/re_cheek_r_human.npy', allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

roi_data_f = []
for i in range(len(raw_data_f)):
    roi_data_f.append(np.average(np.reshape(raw_data_f[i][:14, :, :], (14,-1)), axis=1))
roi_data_f = np.array(roi_data_f)

roi_data_cl = []
for i in range(len(raw_data_cl)):
    roi_data_cl.append(np.average(np.reshape(raw_data_cl[i][:14, :, :], (14,-1)), axis=1))
roi_data_cl = np.array(roi_data_cl)

roi_data_cr = []
for i in range(len(raw_data_cr)):
    roi_data_cr.append(np.average(np.reshape(raw_data_cr[i][:14, :, :], (14,-1)), axis=1))
roi_data_cr = np.array(roi_data_cr)

############# Reference Data ##########################
refer_data = np.load(path + '/../check_data/Reference_Data/input_data9.npy', allow_pickle=True)

for ii in range(np.shape(refer_data)[1]):
    if refer_data[0][ii][0] == 0.14 and refer_data[0][ii][1] == 0.07 and refer_data[0][ii][3] == 0.06:
        ref_data = refer_data[:, ii, 4]

ref_data2 = []
ref_data2.append(ref_data[9])
ref_data2.append(ref_data[12])
ref_data2.append(ref_data[16])
ref_data2.append(ref_data[19])
ref_data2.append(ref_data[25])
ref_data2.append(ref_data[27])
ref_data2.append(ref_data[29])
ref_data2.append(ref_data[31])
ref_data2.append(ref_data[34])
ref_data2.append(ref_data[35])
ref_data2.append(ref_data[44])
ref_data2.append(ref_data[47])
ref_data2.append(ref_data[48])
ref_data2.append(ref_data[49])

###########################################################


wavelength = [490.83, 501.66, 517.64, 529.52, 554.8, 564.59, 574.99, 586.3, 602.75, 609.2, 668.79, 687.2747, 702.84, 710.18]

roi_data_f[0][8] = roi_data_f[0][8]*1.2
roi_data_f[0][9] = roi_data_f[0][9]*1.2
roi_data_f[0][10] = roi_data_f[0][10]*1.2 #  +0.1
roi_data_f[0][11] = roi_data_f[0][11]*1.2 #  +0.1
roi_data_f[0][12] = roi_data_f[0][12]*1.2 #  +0.1
roi_data_f[0][13] = roi_data_f[0][13]*1.2 #  +0.1

plt.figure()
plt.plot(wavelength, roi_data_f[0])
plt.plot(wavelength, np.exp(-(-np.log(roi_data_f[0])+0.85)))
# plt.plot(wavelength, np.exp(-(-np.log(roi_data_cl[0])+0.)))
plt.plot(wavelength, ref_data2)
plt.ylabel('Reflectance')

# plt.figure()
# plt.plot(wavelength, -np.log(roi_data_cl[0]))
# plt.plot(wavelength, -np.log(roi_data_cl[0])+0.4)
# plt.ylabel('Absorption')
plt.show()