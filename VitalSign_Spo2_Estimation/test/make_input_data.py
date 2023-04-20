import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)

file_name = 'TPR4_3'
###############################################
raw_data = np.load(path + '/../check_data/0118_Surface_Ref/{}/re_forehead_human.npy'.format(file_name), allow_pickle=True)
raw_data2 = np.load(path + '/../check_data/0118_Surface_Ref/{}/re_cheek_l_human.npy'.format(file_name), allow_pickle=True)
raw_data3 = np.load(path + '/../check_data/0118_Surface_Ref/{}/re_cheek_r_human.npy'.format(file_name), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

roi_data = []
for i in range(len(raw_data)):
    temp_data = np.average(np.reshape(raw_data[i][:14, :, :], (14, -1)), axis=1)
    for j in range(10):
        temp_data[j] = temp_data[j]
    roi_data.append(temp_data)

roi_data2 = []
for i in range(len(raw_data)):
    temp_data = np.average(np.reshape(raw_data2[i][:14, :, :], (14, -1)), axis=1)
    for j in range(10):
        temp_data[j] = temp_data[j]*0.95
    roi_data2.append(temp_data)

roi_data3 = []
for i in range(len(raw_data)):
    roi_data3.append(np.average(np.reshape(raw_data3[i][:14, :, :], (14,-1)), axis=1))

#######################################

# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_forehead2.npy'.format(file_name), roi_data)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_l2.npy'.format(file_name), roi_data2)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_r2.npy'.format(file_name), roi_data3)
