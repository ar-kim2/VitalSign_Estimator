import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)

file_name1 = 'total_data2'
file_name2_1 = 'TPR8_sh1'
file_name2_2 = 'TPR8_sh2'
file_name3 = 'TPR8_sh'

###############################################
raw_data_f1 = np.load(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name2_1), allow_pickle=True)
raw_data_f2 = np.load(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name2_2), allow_pickle=True)
raw_data_e1 = np.load(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name2_1), allow_pickle=True)
raw_data_e2 = np.load(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name2_2), allow_pickle=True)
raw_data_c1 = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2_1), allow_pickle=True)
raw_data_c2 = np.load(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name2_2), allow_pickle=True)
raw_data_n1 = np.load(path + '/../check_data/{}/{}/re_under_nose_human.npy'.format(file_name1, file_name2_1), allow_pickle=True)
raw_data_n2 = np.load(path + '/../check_data/{}/{}/re_under_nose_human.npy'.format(file_name1, file_name2_2), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

concat_f = np.concatenate((raw_data_f1[:5000], raw_data_f2), axis=0)
concat_e = np.concatenate((raw_data_e1[:5000], raw_data_e2), axis=0)
concat_c = np.concatenate((raw_data_c1[:5000], raw_data_c2), axis=0)
concat_n = np.concatenate((raw_data_n1[:5000], raw_data_n2), axis=0)

np.save(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name3), concat_f)
np.save(path + '/../check_data/{}/{}/re_under_eye_human.npy'.format(file_name1, file_name3), concat_e)
np.save(path + '/../check_data/{}/{}/re_cheek_human.npy'.format(file_name1, file_name3), concat_c)
np.save(path + '/../check_data/{}/{}/re_under_nose_human.npy'.format(file_name1, file_name3), concat_n)


# roi_data = []
# for i in range(len(raw_data)):
#     temp_data = np.average(np.reshape(raw_data[i][:14, :, :], (14, -1)), axis=1)
#     for j in range(10):
#         temp_data[j] = temp_data[j]
#     roi_data.append(temp_data)
#
# roi_data2 = []
# for i in range(len(raw_data)):
#     temp_data = np.average(np.reshape(raw_data2[i][:14, :, :], (14, -1)), axis=1)
#     for j in range(10):
#         temp_data[j] = temp_data[j]*0.95
#     roi_data2.append(temp_data)
#
# roi_data3 = []
# for i in range(len(raw_data)):
#     roi_data3.append(np.average(np.reshape(raw_data3[i][:14, :, :], (14,-1)), axis=1))

#######################################

# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_forehead2.npy'.format(file_name), roi_data)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_l2.npy'.format(file_name), roi_data2)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_r2.npy'.format(file_name), roi_data3)
