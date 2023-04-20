import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)

file_name1 = '0126'
file_name2 = 'TPR3_full'
###############################################
raw_data = np.load(path + '/../check_data/{}/{}/re_forehead_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data2 = np.load(path + '/../check_data/{}/{}/re_cheek_l_human.npy'.format(file_name1, file_name2), allow_pickle=True)
raw_data3 = np.load(path + '/../check_data/{}/{}/re_cheek_r_human.npy'.format(file_name1, file_name2), allow_pickle=True)

############ ROI Detection ############
# roi_data = np.array(raw_data[:, wavelength_idx, 15, 15])

start_idx = 0
end_idx = 100
# Forehead
cali_data = []
roi_data = []
for i in range(start_idx, end_idx):
    temp_data = np.copy(raw_data[i][:14, :, :])
    for j in range(10):
        temp_data[j] = temp_data[j]*1
    for j in range(10, 14):
        temp_data[j] = temp_data[j]*1.28

    cali_data.append(temp_data)
    temp_data2 = np.average(np.reshape(temp_data, (14, -1)), axis=1)
    roi_data.append(temp_data2)
roi_data = np.array(roi_data)

# Cheek L
cali_data2 = []
roi_data2 = []
for i in range(start_idx, end_idx):
    temp_data = np.copy(raw_data2[i][:14, :, :])
    for j in range(10):
        temp_data[j] = temp_data[j]*1
    for j in range(10, 14):
        temp_data[j] = temp_data[j]*1.27

    cali_data2.append(temp_data)
    temp_data2 = np.average(np.reshape(temp_data, (14, -1)), axis=1)
    roi_data2.append(temp_data2)
roi_data2 = np.array(roi_data2)

# Cheek R
cali_data3 = []
roi_data3 = []
for i in range(start_idx, end_idx):
    temp_data = np.copy(raw_data3[i][:14, :, :])
    for j in range(10):
        temp_data[j] = temp_data[j]*1
    for j in range(10, 14):
        temp_data[j] = temp_data[j]*1.24

    cali_data3.append(temp_data)
    temp_data2 = np.average(np.reshape(temp_data, (14, -1)), axis=1)
    roi_data3.append(temp_data2)
roi_data3 = np.array(roi_data3)

#######################################

plt.figure()
plt.title("Visible")
plt.plot(roi_data[:, 7], label='forehead')
plt.plot(roi_data2[:, 7], label='cheek_l')
plt.plot(roi_data3[:, 7], label='cheek_r')
plt.legend()


plt.figure()
plt.title("NIR")
plt.plot(roi_data[:, 11], label='forehead')
plt.plot(roi_data2[:, 11], label='cheek_r')
plt.plot(roi_data3[:, 11], label='cheek_l')
plt.legend()
plt.show()

np.save(path + '/../check_data/{}/{}/re_forehead_human_cali.npy'.format(file_name1, file_name2), cali_data)
np.save(path + '/../check_data/{}/{}/re_cheek_l_human_cali.npy'.format(file_name1, file_name2), cali_data2)
np.save(path + '/../check_data/{}/{}/re_cheek_r_human_cali.npy'.format(file_name1, file_name2), cali_data3)

# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_forehead_2.npy'.format(file_name), roi_data)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_l_2.npy'.format(file_name), roi_data2)
# np.save(path + '/../check_data/0118_Surface_Ref/{}/input_cheek_r_2.npy'.format(file_name), roi_data3)

