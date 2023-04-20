import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import random
import torch.nn as nn

import torch.nn.functional as F
import csv

from scipy.signal import butter, lfilter
from scipy import fftpack
from scipy import interpolate

import copy


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


def calculate_k(x, y, z):
    # x : 851.35  measure idx : 24
    # y : 490.83  measure idx : 0
    # z : 668.79  measure idx : 10

    a = -54.540945783151464
    b = 21.095479254243322
    c = 78.56709080853545
    d =  -4.968144415839242

    t = -(a*x+b*y+c*z+d)/(a+b+c)
    meaure_r_p = x+t
    meaure_g_p = y+t
    meaure_b_p = z+t

    k = x- meaure_r_p

    distance = (((x- meaure_r_p) ** 2) + ((y - meaure_g_p) ** 2) +((z- meaure_b_p) ** 2))/3
    distance = distance**0.5

    # print("Origin point : ", x, ", ",y, ", ",z)
    # print("Covert point : ", meaure_r_p, ", ",meaure_g_p, ", ",meaure_b_p)
    # print("CHECK distance : ", distance, " k : ", k)

    return distance, k

class VitalSign_Feature_mel_thickness(nn.Module):
    def __init__(self):
        super(VitalSign_Feature_mel_thickness, self).__init__()

        self.input_dim = 14

        self.common1 = nn.Linear(self.input_dim, 128)
        self.common2 = nn.Linear(128, 128)
        self.common3 = nn.Linear(128, 128)
        self.common4 = nn.Linear(128, 128)
        self.common5 = nn.Linear(128, 128)

    def forward(self, x):
        x = F.leaky_relu(self.common1(x))
        x = F.leaky_relu(self.common2(x))
        x = F.leaky_relu(self.common3(x))
        x = F.leaky_relu(self.common4(x))
        x = F.leaky_relu(self.common5(x))

        return x

class Classifier(nn.Module):
    def __init__(self, cl_mode):
        super(Classifier, self).__init__()

        self.input_dim = 128

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 128)

        if cl_mode == 'mel':
            self.layer15 = nn.Linear(128, 8)
        elif cl_mode == 'thb':
            self.layer15 = nn.Linear(128, 7)
        elif cl_mode == 'sto':
            self.layer15 = nn.Linear(128, 7)
        elif cl_mode == 'thickness':
            self.layer15 = nn.Linear(128, 3)
        else:
            self.layer15 = nn.Linear(128, 49)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1

class Regression(nn.Module):
    def __init__(self, cl_mode):
        super(Regression, self).__init__()

        if cl_mode == 'mel':
            self.input_dim = 8
        elif cl_mode == 'thb':
            self.input_dim = 7
        elif cl_mode == 'sto':
            self.input_dim = 7
        elif cl_mode == 'thickness':
            self.input_dim = 3
        elif cl_mode == 'fc':
            self.input_dim = 14
        else:
            self.input_dim = 49

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 64)
        self.layer15 = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1


def read_measurement_elapsed_time(file_name):
    path = os.path.dirname(__file__)

    f = open(path + '/check_data/total_data1/{}/time_stamp.csv'.format(file_name), encoding='utf-8')
    rdr = csv.reader(f)

    measure_time = []

    first_time = 0
    for line in rdr:
        if line[0] != 'number' and  line[0] != '':
            str_time = line[1]
            hour = int(str_time[-15:-13])
            min = int(str_time[-12:-10])
            sec = int(str_time[-9:-7])
            ms = int(str_time[-6:-1])

            total_time = hour * 60 * 60 * 100000 + min * 60 * 100000 + sec * 100000 + ms

            # if line[0] == '0':
            #     first_time = total_time

            if first_time == 0:
                first_time = total_time

            measure_time.append((total_time - first_time) / 100000)

            if measure_time[-1] < 0:
                print("min {} sec {} ms {} idx {}".format(min, sec, ms, line))

    f.close()

    measure_time = np.array(measure_time)

    return measure_time

def read_pulse_data(file_name):
    path = os.path.dirname(__file__)

    f = open(path + '/check_data/total_data1/{}/pulse_sto.csv'.format(file_name), encoding='utf-8')
    rdr = csv.reader(f)

    pulse_list = []

    for line in rdr:
        if line[3] != 'PULSE' and line[3] !='':
            str_time = line[3]
            pulse = int(str_time)
            pulse_list.append(pulse)
    f.close()

    pulse_list = np.array(pulse_list)

    pulse_time = []
    for i in range(len(pulse_list)):
        pulse_time.append(i)

    pulse_time = np.array(pulse_time)

    return pulse_list, pulse_time

def read_spo2_data(file_name):
    path = os.path.dirname(__file__)

    f = open(path + '/check_data/total_data1/{}/pulse_sto.csv'.format(file_name), encoding='utf-8')
    rdr = csv.reader(f)

    spo2_list = []

    for line in rdr:
        if line[2] != 'SPO2' and line[2] !='':
            str_time = line[2]
            spo2 = int(str_time)
            spo2_list.append(spo2)
    f.close()

    spo2_list = np.array(spo2_list)

    sp_time = []
    for i in range(len(spo2_list)):
        sp_time.append(i)

    spo2_time = np.array(sp_time)

    return spo2_list, spo2_time

def read_ppg_data(file_name):
    path = os.path.dirname(__file__)
    f = open(path + '/check_data/total_data1/{}/ppg_wave.csv'.format(file_name), encoding='utf-8')

    rdr = csv.reader(f)

    ppg_wave = []

    for line in rdr:

        if line[0] != 'Wave':
            ppg_wave.append(int(line[0]))

    f.close()

    ppg_wave = np.array(ppg_wave)

    ppg_time = []
    for i in range(len(ppg_wave)):
        ppg_time.append(i * (1 / 60))

    ppg_time = np.array(ppg_time)

    return ppg_wave, ppg_time



class ViatalSignDataset_pulse_fft(data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode

        if mode == "train":
            input_data, gt_data = self.read_vitalsign_dataset(name='train')
        else:
            input_data, gt_data = self.read_vitalsign_dataset(name='test')

        self.input_data, self.gt_data = input_data, gt_data

    def __getitem__(self, index):
        input = self.input_data[index]
        gt = self.gt_data[index]

        print(" get item check ")
        input = torch.FloatTensor(input)
        gt = torch.FloatTensor(gt)

        return input, gt

    def __len__(self):
        return len(self.input_data)

    def read_vitalsign_dataset(self, name):
        '''
        Read Data
        '''

        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path+'/check_data/total_data1/')
        fileNameList = fileNameList[:1]

        input_data = []
        gt_data = []

        for fn in fileNameList:
            print("Filename : ", fn)
            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)

            roi_data = []
            for i in range(len(temp_raw_data)):
                temp_data = np.average(temp_raw_data[i][6][:, :])
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            measure_time = read_measurement_elapsed_time(fn)
            gt_pulse_list = read_pulse_data(fn)

            time_window = 10

            temp_input_seq = []
            temp_gt = []

            if name == 'train':
                time_start_idx = 10
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160

            f_linear = interpolate.interp1d(measure_time[:len(roi_data)], roi_data, kind='linear')
            # print("CHECK mearue time end : ", measure_time[len(roi_data)])

            check_start_cnt = 0

            # Training data use first 2 min
            # For using bandpass filter, start 10s
            for ti in range(time_start_idx, time_end_idx-time_window):
                start_idx = int(np.where(ti <= measure_time)[0][0])
                end_idx = int(np.where(measure_time <= (ti + time_window))[0][-1])

                # f_linear = interpolate.interp1d(measure_time[start_idx - 400:end_idx + 10],
                #                                 roi_data[start_idx - 400:end_idx + 10], kind='linear')

                fs = 48
                sample_tiem = np.arange(ti-6, ti+time_window, (1/48))
                # print("sample time shape : ", np.shape(sample_tiem))
                # print("sample time : ", sample_tiem)

                sample_ppg = f_linear(sample_tiem)

                # print("sample ppg shape : ", np.shape(sample_ppg))
                # print("sample ppg : ", sample_ppg)

                # start_idx = int(np.where(ti <= measure_time)[0][0])
                # end_idx = int(np.where(measure_time <= (ti + time_window))[0][-1])
                #
                # # Calculate Fps
                # sub_time = []
                # for i in range(start_idx, end_idx):
                #     sub_time.append(measure_time[i] - measure_time[i - 1])
                #
                # fs = 1 / np.average(sub_time)

                # bp_data = np.copy(roi_data[start_idx - 400:end_idx])
                bp_data = np.copy(sample_ppg)

                if fs < 10:
                    fs = 10

                bp_data2 = butter_bandpass_filter(bp_data, 0.5, 3, fs)

                if len(bp_data2) < 400:
                    bp_data2 = bp_data2
                else:
                    # bp_data2 = bp_data2[400:]
                    bp_data2 = bp_data2[288:]

                sig_fft = fftpack.fft(bp_data2)
                power = np.abs(sig_fft)
                sample_freq = fftpack.fftfreq(len(bp_data2), d=(1 / fs))
                mask = (sample_freq > 0.5) & (sample_freq < 3)

                # temp_power = np.array(power[mask])
                # max_power = np.round(np.max(temp_power), 3)
                # max_arg = np.argmax(temp_power)
                # max_freq = np.round(sample_freq[mask][max_arg], 3)

                # print("CHECK len power ", len(power[mask]))
                # print("CHECK sample_freq ", sample_freq[mask])
                # input power shape (25,)
                # if len(power[mask]) != 24:
                #     temp_input_seq.append(temp_input_seq[-1])
                # else:
                temp_input_seq.append(power[mask])

                # temp_input_seq.append([max_freq])
                temp_gt.append(gt_pulse_list[ti + time_window])

                check_start_cnt = check_start_cnt+1

                if check_start_cnt > 5:
                    input_data.append(copy.deepcopy(temp_input_seq))
                    gt_data.append(copy.deepcopy(temp_gt))


        # input_data = np.array(input_data, dtype=np.float32)
        # gt_data = np.array(gt_data, dtype=np.float32)

        print("CHECK train data shape2 : ", np.shape(input_data))
        print("CHECK train gt data shape : ", np.shape(gt_data))

        return input_data, gt_data


class ViatalSignDataset_triplet(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, ppg_label, spo2_label = self.read_vitalsign_dataset(name='train')
        else:
            reflect_list, ppg_label, spo2_label = self.read_vitalsign_dataset(name='test')

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_reflect_list= []
        for i in range(len(reflect_list)):
            if i < mv_window:
                mf_reflect_list.append(reflect_list[i])
            else:
                mf_reflect_list.append(list(np.average(reflect_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob2 = []

        mf_mel_value = []
        for i in range(len(mel_value)):
            if i < mv_window:
                mf_mel_value.append(mel_value[i])
            else:
                mf_mel_value.append(list(np.average(mel_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_mel_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.01) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.03) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.05) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.07) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.09) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.11) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.13) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.15) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            mel_prob2.append(m_i3)

        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob2 = []

        mf_thickness_value = []
        for i in range(len(thickness_value)):
            if i < mv_window:
                mf_thickness_value.append(thickness_value[i])
            else:
                mf_thickness_value.append(list(np.average(thickness_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_thickness_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            thickness_prob2.append(th_i3)

        thickness_prob = thickness_prob.detach().cpu().numpy()

        reflect_list = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((reflect_list, mel_prob2, thickness_prob2), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        self.ref_list = np.array(reflect_list, dtype=np.float32)
        self.total_label = ppg_label*10 + spo2_label

        # self.positive_list = []
        # self.negative_list = []
        #
        # for i in range(len(self.total_label)):
        #     if i % 10000 == 0:
        #         print("Data load 4 ", i)
        #
        #     totall = self.total_label[i]
        #
        #     negative_idx = np.where(self.total_label != totall)
        #     positive_idx = np.where(self.total_label == totall)
        #
        #     self.positive_list.append(positive_idx[0])
        #     self.negative_list.append(negative_idx[0])

    def __getitem__(self, index):
        anchor = self.ref_list[index]
        totall = self.total_label[index]

        # p_idx = self.positive_list[index][torch.randint(len(self.positive_list[index]), (1,))]
        # positive = self.ref_list[p_idx]
        #
        # n_idx = self.negative_list[index][torch.randint(len(self.negative_list[index]), (1,))]
        # negative = self.ref_list[n_idx]

        positive_idx = np.where(self.total_label == totall)
        negative_idx = np.where(self.total_label != totall)

        positive = self.ref_list[random.choice(positive_idx[0])]
        negative = self.ref_list[random.choice(negative_idx[0])]

        return anchor, positive, negative, totall

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        for fn in fileNameList:
            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn), allow_pickle=True)
            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            if name == 'train':
                time_start_idx = 0
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160

            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)

            # print("CHECK input data shape: ", np.shape(input_data))
            # print("CHECK ppg data shape: ", np.shape(gt_ppg_data))
            # print("CHECK spo2 data shape: ", np.shape(gt_spo2_data))

        reflect_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            reflect_list.append(temp_list)

        reflect_list = np.array(reflect_list, dtype=np.float32)

        ppg_label = []
        spo2_label = []

        for d_idx in range(len(gt_ppg_data)):
            if gt_ppg_data[d_idx] > 0 and gt_ppg_data[d_idx] <= 10:
                ppg_i = 0
            elif gt_ppg_data[d_idx] > 10 and gt_ppg_data[d_idx] <= 20:
                ppg_i = 1
            elif gt_ppg_data[d_idx] > 20 and gt_ppg_data[d_idx] <= 30:
                ppg_i = 2
            elif gt_ppg_data[d_idx] > 30 and gt_ppg_data[d_idx] <= 40:
                ppg_i = 3
            elif gt_ppg_data[d_idx] > 40 and gt_ppg_data[d_idx] <= 50:
                ppg_i = 4
            elif gt_ppg_data[d_idx] > 50 and gt_ppg_data[d_idx] <= 60:
                ppg_i = 5
            elif gt_ppg_data[d_idx] > 60 and gt_ppg_data[d_idx] <= 70:
                ppg_i = 6
            elif gt_ppg_data[d_idx] > 70 and gt_ppg_data[d_idx] <= 80:
                ppg_i = 7
            elif gt_ppg_data[d_idx] > 80 and gt_ppg_data[d_idx] <= 90:
                ppg_i = 8
            elif gt_ppg_data[d_idx] > 90 and gt_ppg_data[d_idx] <= 100:
                ppg_i = 9
            elif gt_ppg_data[d_idx] > 100 and gt_ppg_data[d_idx] <= 110:
                ppg_i = 10
            elif gt_ppg_data[d_idx] > 110 and gt_ppg_data[d_idx] <= 120:
                ppg_i = 11
            elif gt_ppg_data[d_idx] > 120:
                ppg_i = 12

            ppg_label.append(ppg_i)

            if gt_spo2_data[d_idx] <= 91:
                sp_i = 0
            elif gt_spo2_data[d_idx] > 91 and gt_spo2_data[d_idx] <= 92:
                sp_i = 1
            elif gt_spo2_data[d_idx] > 92 and gt_spo2_data[d_idx] <= 93:
                sp_i = 2
            elif gt_spo2_data[d_idx] > 93 and gt_spo2_data[d_idx] <= 94:
                sp_i = 3
            elif gt_spo2_data[d_idx] > 94 and gt_spo2_data[d_idx] <= 95:
                sp_i = 4
            elif gt_spo2_data[d_idx] > 95 and gt_spo2_data[d_idx] <= 96:
                sp_i = 5
            elif gt_spo2_data[d_idx] > 96 and gt_spo2_data[d_idx] <= 97:
                sp_i = 6
            elif gt_spo2_data[d_idx] > 97 and gt_spo2_data[d_idx] <= 98:
                sp_i = 7
            elif gt_spo2_data[d_idx] > 98 and gt_spo2_data[d_idx] <= 99:
                sp_i = 8
            elif gt_spo2_data[d_idx] > 99:
                sp_i = 9

            spo2_label.append(sp_i)

        ppg_label = np.array(ppg_label, dtype=np.float32)
        spo2_label = np.array(spo2_label, dtype=np.float32)

        return reflect_list, ppg_label, spo2_label



class ViatalSignDataset_class(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, comb_label = self.read_vitalsign_dataset(name='train')
        else:
            reflect_list, comb_label = self.read_vitalsign_dataset(name='test')

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_reflect_list= []
        for i in range(len(reflect_list)):
            if i < mv_window:
                mf_reflect_list.append(reflect_list[i])
            else:
                mf_reflect_list.append(list(np.average(reflect_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob2 = []

        mf_mel_value = []
        for i in range(len(mel_value)):
            if i < mv_window:
                mf_mel_value.append(mel_value[i])
            else:
                mf_mel_value.append(list(np.average(mel_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_mel_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.01) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.03) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.05) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.07) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.09) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.11) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.13) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.15) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            mel_prob2.append(m_i3)

        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob2 = []

        mf_thickness_value = []
        for i in range(len(thickness_value)):
            if i < mv_window:
                mf_thickness_value.append(thickness_value[i])
            else:
                mf_thickness_value.append(list(np.average(thickness_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_thickness_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            thickness_prob2.append(th_i3)

        thickness_prob = thickness_prob.detach().cpu().numpy()

        reflect_list = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((reflect_list, mel_prob2, thickness_prob2), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.comb_label = torch.FloatTensor(comb_label).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.comb_label = torch.FloatTensor(comb_label)

        # self.positive_list = []
        # self.negative_list = []
        #
        # for i in range(len(self.total_label)):
        #     if i % 10000 == 0:
        #         print("Data load 4 ", i)
        #
        #     totall = self.total_label[i]
        #
        #     negative_idx = np.where(self.total_label != totall)
        #     positive_idx = np.where(self.total_label == totall)
        #
        #     self.positive_list.append(positive_idx[0])
        #     self.negative_list.append(negative_idx[0])

    def __getitem__(self, index):
        ref = self.ref_list[index]
        comb_label = self.comb_label[index]

        return ref, comb_label

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        for fn in fileNameList:
            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn), allow_pickle=True)
            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            if name == 'train':
                time_start_idx = 0
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160

            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)

            # print("CHECK input data shape: ", np.shape(input_data))
            # print("CHECK ppg data shape: ", np.shape(gt_ppg_data))
            # print("CHECK spo2 data shape: ", np.shape(gt_spo2_data))

        reflect_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            reflect_list.append(temp_list)

        reflect_list = np.array(reflect_list, dtype=np.float32)

        combination_label3 = []

        for d_idx in range(len(gt_ppg_data)):
            g_p = []

            g_sig = np.array([[5 ** 2, 0], [0, 0.5 ** 2]])

            g_x = np.array([gt_ppg_data[d_idx], gt_spo2_data[d_idx]])

            g_ppg_u = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
            g_spo2_u = [90.5, 91.5, 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 98.5, 99.5]

            for g_t in range(len(g_ppg_u)):
                for g_s in range(len(g_spo2_u)):
                    g_u = np.array([g_ppg_u[g_t], g_spo2_u[g_s]])
                    g_p.append(np.exp(-(1/2)*np.dot(np.dot((g_x-g_u).T, np.linalg.inv(g_sig)), (g_x-g_u))) / ((((2*np.pi)**2) * np.linalg.det(g_sig))**(1/2)))

            st_tb_i3 = []
            for gp_i in range(len(g_p)):
                st_tb_i3.append(g_p[gp_i] / np.sum(g_p))

            combination_label3.append(st_tb_i3)

        combination_label3 = np.array(combination_label3, dtype=np.float32)

        return reflect_list, combination_label3


class ViatalSignDataset_regression(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu

        if mode == "train":
            reflect_list, gt_ppg_data, gt_spo2_data = self.read_vitalsign_dataset(name='train')
        else:
            reflect_list, gt_ppg_data, gt_spo2_data = self.read_vitalsign_dataset(name='test')

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_reflect_list= []
        for i in range(len(reflect_list)):
            if i < mv_window:
                mf_reflect_list.append(reflect_list[i])
            else:
                mf_reflect_list.append(list(np.average(reflect_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob2 = []

        mf_mel_value = []
        for i in range(len(mel_value)):
            if i < mv_window:
                mf_mel_value.append(mel_value[i])
            else:
                mf_mel_value.append(list(np.average(mel_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_mel_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.01) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.03) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.05) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.07) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.09) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.11) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.13) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_mel_value[d_idx][0] - 0.15) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            m_i3 = []
            for gp_i in range(len(g_p)):
                m_i3.append(g_p[gp_i]/np.sum(g_p))

            mel_prob2.append(m_i3)

        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_reflect_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob2 = []

        mf_thickness_value = []
        for i in range(len(thickness_value)):
            if i < mv_window:
                mf_thickness_value.append(thickness_value[i])
            else:
                mf_thickness_value.append(list(np.average(thickness_value[i - mv_window:i, :], axis=0)))

        for d_idx in range(len(mf_thickness_value)):
            g_p = []
            g_sig = 0.005

            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.025) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.045) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))
            g_p.append(np.exp(-(mf_thickness_value[d_idx][0] - 0.065) ** 2 / (2 * g_sig**2)) / (np.sqrt(2 * np.pi * g_sig**2)))

            th_i3 = []
            for gp_i in range(len(g_p)):
                th_i3.append(g_p[gp_i] / np.sum(g_p))

            thickness_prob2.append(th_i3)

        thickness_prob = thickness_prob.detach().cpu().numpy()

        reflect_list = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((reflect_list, mel_prob2, thickness_prob2), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        if self.use_gpu == True:
            self.ref_list = torch.FloatTensor(reflect_list).to('cuda')
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data).to('cuda')
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data).to('cuda')
        else:
            self.ref_list = torch.FloatTensor(reflect_list)
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data)
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data)

    def __getitem__(self, index):
        ref = self.ref_list[index]
        ppg_data = self.gt_ppg_data[index]
        spo2_data = self.gt_spo2_data[index]

        return ref, ppg_data, spo2_data

    def __len__(self):
        return len(self.ref_list)

    def read_vitalsign_dataset(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        for fn in fileNameList:
            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn), allow_pickle=True)
            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            if name == 'train':
                time_start_idx = 0
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160

            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)

            # print("CHECK input data shape: ", np.shape(input_data))
            # print("CHECK ppg data shape: ", np.shape(gt_ppg_data))
            # print("CHECK spo2 data shape: ", np.shape(gt_spo2_data))

        reflect_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            reflect_list.append(temp_list)

        reflect_list = np.array(reflect_list, dtype=np.float32)
        gt_ppg_data = np.array(gt_ppg_data, dtype=np.float32)
        gt_spo2_data = np.array(gt_spo2_data, dtype=np.float32)

        return reflect_list, gt_ppg_data, gt_spo2_data

# Personal data/ Training 2min, Test 1min
class ViatalSignDataset_ppg_lstm(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, seq_len = 100, roi = 'forehead'):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu
        self.seq_len = seq_len

        if mode == "train":
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='train', roi=roi)
        else:
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='test', roi=roi)

        # Mel, Thickness 추정 모델 Load
        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        # 멜라닌과 Thickness 추정을 위해 Absorbance data에 Moving Average Filter 적용
        mv_window = 30

        mf_absorption_list= []
        for i in range(len(absorption_list)):
            if i < mv_window:
                mf_absorption_list.append(absorption_list[i])
            else:
                mf_absorption_list.append(list(np.average(absorption_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob = thickness_prob.detach().cpu().numpy()

        # Absorbance와 Melanin, Thickness 확률 분포 추정 결과 Concatenate
        absorption_list_concat = np.concatenate((absorption_list, mel_prob, thickness_prob), axis=1)
        reflect_list_ = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        sequence_absorption = []
        sequence_absorption_concat = []
        sequence_reflectance = []
        sequence_ppg = []
        sequence_spo2 = []
        sequence_pulse = []

        mel_value_list = []
        thickness_value_list = []

        file_start = 0
        file_check_idx = 0

        # 입력 및 GT data들을 Sequence data로 변환
        for seq_i in range(1, len(absorption_list_concat), 3):
            if seq_i >= sample_time_list[file_check_idx]:
                file_start = sample_time_list[file_check_idx]
                file_check_idx = file_check_idx+1

            if (seq_i - file_start) < self.seq_len:
                continue
            else:

                if self.use_gpu == True:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption_concat.append(absorption_list_concat[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i-(self.seq_len-1):seq_i+1])

                    mel_value_list.append(mel_value[seq_i + 1])
                    thickness_value_list.append(thickness_value[seq_i + 1])

                else:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption_concat.append(absorption_list_concat[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i - (self.seq_len - 1):seq_i + 1])

                    mel_value_list.append(mel_value[seq_i + 1])
                    thickness_value_list.append(thickness_value[seq_i + 1])

        if self.use_gpu == True:
            self.absorption_list = torch.FloatTensor(absorption_list).to('cuda')
            self.absorption_concat_list = torch.FloatTensor(absorption_list_concat).to('cuda')
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data).to('cuda')
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data).to('cuda')
            self.sequence_absorption = torch.FloatTensor(sequence_absorption).to('cuda')
            self.sequence_absorption_concat = torch.FloatTensor(sequence_absorption_concat).to('cuda')
            self.sequence_reflectance = torch.FloatTensor(sequence_reflectance).to('cuda')
            self.sequence_ppg = torch.FloatTensor(sequence_ppg).to('cuda')
            self.sequence_spo2 = torch.FloatTensor(sequence_spo2).to('cuda')
            self.sequence_pulse = torch.FloatTensor(sequence_pulse).to('cuda')

            self.mel_value_list = mel_value_list
            self.thickness_value_list = thickness_value_list

        else:
            self.absorption_list = torch.FloatTensor(absorption_list)
            self.absorption_concat_list = torch.FloatTensor(absorption_list_concat)
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data)
            self.gt_spo2_data = torch.Tensor(gt_spo2_data)

            self.mel_value_list = mel_value_list
            self.thickness_value_list = thickness_value_list

            self.sequence_absorption = torch.Tensor(sequence_absorption)
            self.sequence_absorption_concat = torch.Tensor(sequence_absorption_concat)
            self.sequence_reflectance = torch.Tensor(sequence_reflectance)
            self.sequence_ppg = torch.Tensor(sequence_ppg)
            self.sequence_spo2 = torch.Tensor(sequence_spo2)
            self.sequence_pulse = torch.Tensor(sequence_pulse)

    def __getitem__(self, index):
        # abs = self.absorption_list[index]
        sequence_absorption = self.sequence_absorption[index]
        sequence_absorption_concat = self.sequence_absorption_concat[index]
        sequence_reflectance = self.sequence_reflectance[index]
        ppg_data = self.sequence_ppg[index]
        spo2_data = self.sequence_spo2[index]
        pulse_data = self.sequence_pulse[index]

        mel_value = self.mel_value_list[index]
        thickness_value = self.thickness_value_list[index]

        return sequence_absorption, sequence_absorption_concat, sequence_reflectance, ppg_data, spo2_data, pulse_data, mel_value, thickness_value

    def __len__(self):
        return len(self.sequence_absorption)

    def read_vitalsign_dataset(self, name, roi):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        # 전체 Dataset에서 초기 2분은 Training, 이후 1분은 Test로 사용하는 경우
        fileNameList = os.listdir(path + '/check_data/total_data1/')

        # 일부 파일의 이후 1분 data로 Test하는 경우
        # fileNameList = ['TPR1_ar', 'TPR6_aron']

        set_fps = 30

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        sample_time_list = []

        for fn in fileNameList:
            # Load Reflectance data, Reflectance Shape (time_stamp, spectral_band, x_axis, y_axis)
            print(fn)
            if roi == 'forehead':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'ueye':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_eye_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'cheek':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_cheek_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'unose':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn),
                                        allow_pickle=True)

            # Load Time, Ground Truth PPG, Spo2, Pulse data
            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)
            gt_pulse, gt_pulse_time = read_pulse_data(fn)

            # ROI 영역의 평균값 계산
            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            # Train data는 초기 2분, Test data는 이후 1분
            if name == 'train':
                time_start_idx = 0
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160
                
            # 해당 시간에 해당하는 List의 index를 찾음.
            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            # Interpolation을 통해 set_fps(30)에 해당하는 reflectance data를 가져옴.
            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / set_fps))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            # Ground Truth에 해당하는 data도 Interpolation을 통해 동일한 set_fps에 해당하는 값으로 변환
            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / set_fps))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            f_linear_pulse = interpolate.interp1d(gt_pulse_time, gt_pulse, kind='linear')
            sample_pulse = f_linear_pulse(sample_tiem)

            if len(sample_time_list) == 0:
                sample_time_list.append(len(sample_tiem))
            else:
                sample_time_list.append(sample_time_list[-1]+len(sample_tiem))

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
                gt_pulse_data = sample_pulse
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)
                gt_pulse_data = np.concatenate([gt_pulse_data, sample_pulse], axis=0)

        # Reflectance data에서 Shading을 제거한 뒤 Absorbance로 변환
        
        absorption_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            absorption_list.append(temp_list)

        absorption_list = np.array(absorption_list, dtype=np.float32)
        gt_ppg_data = np.array(gt_ppg_data, dtype=np.float32)
        gt_spo2_data = np.array(gt_spo2_data, dtype=np.float32)
        gt_pulse_data = np.array(gt_pulse_data, dtype=np.float32)

        # input_data: Reflectance data, absorption_list: absorbance data
        return input_data, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list

# Personal Data, Trainin data 9 person, Test 3 Person
class ViatalSignDataset_ppg_lstm2(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, seq_len = 100, roi = 'forehead', test_name = ''):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu
        self.seq_len = seq_len

        if mode == "train":
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='train', roi=roi, test_name=test_name)
        else:
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='test', roi=roi, test_name=test_name)

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_absorption_list= []
        for i in range(len(absorption_list)):
            if i < mv_window:
                mf_absorption_list.append(absorption_list[i])
            else:
                mf_absorption_list.append(list(np.average(absorption_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob = thickness_prob.detach().cpu().numpy()

        absorption_list_concat = np.concatenate((absorption_list, mel_prob, thickness_prob), axis=1)
        reflect_list_ = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        sequence_absorption = []
        sequence_absorption_concat = []
        sequence_reflectance = []
        sequence_ppg = []
        sequence_spo2 = []
        sequence_pulse = []
        file_start = 0
        file_check_idx = 0

        for seq_i in range(1, len(absorption_list_concat), 3):
            if seq_i >= sample_time_list[file_check_idx]:
                file_start = sample_time_list[file_check_idx]
                file_check_idx = file_check_idx+1

            if (seq_i - file_start) < self.seq_len:
                continue
            else:
                if self.use_gpu == True:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption_concat.append(absorption_list_concat[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i-(self.seq_len-1):seq_i+1])

                else:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption_concat.append(absorption_list_concat[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i - (self.seq_len - 1):seq_i + 1])

        if self.use_gpu == True:
            self.absorption_list = torch.FloatTensor(absorption_list).to('cuda')
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data).to('cuda')
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data).to('cuda')

            self.sequence_absorption = torch.FloatTensor(sequence_absorption).to('cuda')
            self.sequence_absorption_concat = torch.FloatTensor(sequence_absorption_concat).to('cuda')
            self.sequence_reflectance = torch.FloatTensor(sequence_reflectance).to('cuda')
            self.sequence_ppg = torch.FloatTensor(sequence_ppg).to('cuda')
            self.sequence_spo2 = torch.FloatTensor(sequence_spo2).to('cuda')
            self.sequence_pulse = torch.FloatTensor(sequence_pulse).to('cuda')
        else:
            self.absorption_list = torch.FloatTensor(absorption_list)
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data)
            self.gt_spo2_data = torch.Tensor(gt_spo2_data)

            self.sequence_absorption = torch.Tensor(sequence_absorption)
            self.sequence_absorption_concat = torch.Tensor(sequence_absorption_concat)
            self.sequence_reflectance = torch.Tensor(sequence_reflectance)
            self.sequence_ppg = torch.Tensor(sequence_ppg)
            self.sequence_spo2 = torch.Tensor(sequence_spo2)
            self.sequence_pulse = torch.Tensor(sequence_pulse)

    def __getitem__(self, index):
        # abs = self.absorption_list[index]
        sequence_absorption = self.sequence_absorption[index]
        sequence_absorption_concat = self.sequence_absorption_concat[index]
        sequence_reflectance = self.sequence_reflectance[index]
        ppg_data = self.sequence_ppg[index]
        spo2_data = self.sequence_spo2[index]
        pulse_data = self.sequence_pulse[index]

        return sequence_absorption, sequence_absorption_concat, sequence_reflectance, ppg_data, spo2_data, pulse_data

    def __len__(self):
        return len(self.sequence_absorption)

    def read_vitalsign_dataset(self, name, roi, test_name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        set_fps = 30

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        sample_time_list = []


        for fn in fileNameList:
            if name == 'train':
                print("Trainin")
                if fn.__contains__(test_name):
                    continue
                else:
                    print(fn)
            else:
                print("test")
                if (fn.__contains__(test_name) == False):
                    continue
                else:
                    print(fn)

            if roi == 'forehead':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'ueye':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_eye_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'cheek':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_cheek_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'unose':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn),
                                        allow_pickle=True)

            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)

            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)
            gt_pulse, gt_pulse_time = read_pulse_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            time_start_idx = 0
            time_end_idx = 160

            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / set_fps))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / set_fps))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            f_linear_pulse = interpolate.interp1d(gt_pulse_time, gt_pulse, kind='linear')
            sample_pulse = f_linear_pulse(sample_tiem)

            if len(sample_time_list) == 0:
                sample_time_list.append(len(sample_tiem))
            else:
                sample_time_list.append(sample_time_list[-1]+len(sample_tiem))

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
                gt_pulse_data = sample_pulse
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)
                gt_pulse_data = np.concatenate([gt_pulse_data, sample_pulse], axis=0)

        absorption_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            absorption_list.append(temp_list)

        absorption_list = np.array(absorption_list, dtype=np.float32)
        gt_ppg_data = np.array(gt_ppg_data, dtype=np.float32)
        gt_spo2_data = np.array(gt_spo2_data, dtype=np.float32)
        gt_pulse_data = np.array(gt_pulse_data, dtype=np.float32)

        return input_data, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list



# keep breath model
class ViatalSignDataset_ppg_lstm3(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, seq_len = 100, roi = 'forehead'):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu
        self.seq_len = seq_len

        if mode == "train":
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='train', roi=roi)
        else:
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='test', roi=roi)

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_absorption_list= []
        for i in range(len(absorption_list)):
            if i < mv_window:
                mf_absorption_list.append(absorption_list[i])
            else:
                mf_absorption_list.append(list(np.average(absorption_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob = thickness_prob.detach().cpu().numpy()

        absorption_list = np.concatenate((absorption_list, mel_prob, thickness_prob), axis=1)
        reflect_list_ = np.concatenate((reflect_list, mel_prob, thickness_prob), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        print("CHECK absorption shape 11: ", np.shape(absorption_list))

        sequence_absorption = []
        sequence_reflectance = []
        sequence_ppg = []
        sequence_spo2 = []
        sequence_pulse = []
        file_start = 0
        file_check_idx = 0

        # gt_ppg_data = np.delete(gt_ppg_data, 0)
        # gt_spo2_data = np.delete(gt_spo2_data, 0)

        for seq_i in range(1, len(absorption_list), 5):
            if seq_i >= sample_time_list[file_check_idx]:
                file_start = sample_time_list[file_check_idx]
                file_check_idx = file_check_idx+1
                # gt_ppg_data = np.delete(gt_ppg_data, seq_i)
                # gt_spo2_data = np.delete(gt_spo2_data, seq_i)
            # elif (seq_i - file_start) > 3600:
            #     file_start = seq_i

            if (seq_i - file_start) < self.seq_len:
                continue
            else:
                # if self.use_gpu == True:
                #     sequence_absorption.append(torch.FloatTensor(absorption_list[file_start:seq_i+1]).to('cuda'))
                #     sequence_reflectance.append(torch.FloatTensor(reflect_list[file_start:seq_i+1]).to('cuda'))
                #     sequence_ppg.append(torch.FloatTensor(gt_ppg_data[file_start:seq_i+1]).to('cuda'))
                #     sequence_spo2.append(torch.FloatTensor(gt_spo2_data[file_start:seq_i+1]).to('cuda'))
                # else:
                #     sequence_absorption.append(torch.FloatTensor(absorption_list[file_start:seq_i+1]))
                #     sequence_reflectance.append(torch.FloatTensor(reflect_list[file_start:seq_i+1]))
                #     sequence_ppg.append(torch.FloatTensor(gt_ppg_data[file_start:seq_i+1]))
                #     sequence_spo2.append(torch.FloatTensor(gt_spo2_data[file_start:seq_i+1]))

                if self.use_gpu == True:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i-(self.seq_len-1):seq_i+1])

                    # print("CHECK sequence_absorption : ", np.shape(sequence_absorption))
                    #
                    # plt.figure()
                    # plt.plot(sequence_ppg[-1]/200)
                    # plt.plot((np.array(sequence_reflectance)[-1, :, 6]*100)-23)
                    # plt.show()


                else:
                    sequence_absorption.append(absorption_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i - (self.seq_len - 1):seq_i + 1])

                # print("CHECK sequence_absorption shape : ", np.shape(sequence_absorption))

        # self.sequence_absorption = sequence_absorption
        # self.sequence_reflectance = sequence_reflectance
        # self.sequence_ppg = sequence_ppg
        # self.sequence_spo2 = sequence_spo2

        print("CHECK absorption shape22: ", len(sequence_absorption))

        if self.use_gpu == True:
            self.absorption_list = torch.FloatTensor(absorption_list).to('cuda')
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data).to('cuda')
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data).to('cuda')

            self.sequence_absorption = torch.FloatTensor(sequence_absorption).to('cuda')
            self.sequence_reflectance = torch.FloatTensor(sequence_reflectance).to('cuda')
            self.sequence_ppg = torch.FloatTensor(sequence_ppg).to('cuda')
            self.sequence_spo2 = torch.FloatTensor(sequence_spo2).to('cuda')
            self.sequence_pulse = torch.FloatTensor(sequence_pulse).to('cuda')
        else:
            self.absorption_list = torch.FloatTensor(absorption_list)
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data)
            self.gt_spo2_data = torch.Tensor(gt_spo2_data)

            self.sequence_absorption = torch.Tensor(sequence_absorption)
            self.sequence_reflectance = torch.Tensor(sequence_reflectance)
            self.sequence_ppg = torch.Tensor(sequence_ppg)
            self.sequence_spo2 = torch.Tensor(sequence_spo2)
            self.sequence_pulse = torch.Tensor(sequence_pulse)

    def __getitem__(self, index):
        # abs = self.absorption_list[index]
        sequence_absorption = self.sequence_absorption[index]
        sequence_reflectance = self.sequence_reflectance[index]
        ppg_data = self.sequence_ppg[index]
        spo2_data = self.sequence_spo2[index]
        pulse_data = self.sequence_pulse[index]

        # sequence_absorption = torch.FloatTensor(sequence_absorption)
        # sequence_reflectance = torch.FloatTensor(sequence_reflectance)
        # ppg_data = torch.FloatTensor(ppg_data)
        # spo2_data = torch.FloatTensor(spo2_data)

        return sequence_absorption, sequence_reflectance, ppg_data, spo2_data, pulse_data

    def __len__(self):
        return len(self.sequence_absorption)

    def read_vitalsign_dataset(self, name, roi):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        sample_time_list = []

        for fn in fileNameList:
            if name == 'train':
                print("Train")
                if fn[-1] != '1':
                    continue
                else:
                    print(fn)
            else:
                print("test")
                if fn[-1] != '2':
                    continue
                else:
                    print(fn)

            if roi == 'forehead':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'ueye':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_eye_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'cheek':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_cheek_human.npy'.format(fn),
                                        allow_pickle=True)
            elif roi == 'unose':
                temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_under_nose_human.npy'.format(fn),
                                        allow_pickle=True)

            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)

            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)
            gt_pulse, gt_pulse_time = read_pulse_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            time_start_idx = 0
            if len(roi_data) < len(measure_time):
                time_end_idx = int(measure_time[len(roi_data)])-3
            else:
                time_end_idx = int(measure_time[-1]) - 3


            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            print("measure time ", measure_time)
            print("measure time 22 ", measure_time[-1])
            print("roi_data len ", len(roi_data))
            print("roi_data len ", len(measure_time))

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            f_linear_pulse = interpolate.interp1d(gt_pulse_time, gt_pulse, kind='linear')
            sample_pulse = f_linear_pulse(sample_tiem)

            if len(sample_time_list) == 0:
                sample_time_list.append(len(sample_tiem))
            else:
                sample_time_list.append(sample_time_list[-1]+len(sample_tiem))

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
                gt_pulse_data = sample_pulse
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)
                gt_pulse_data = np.concatenate([gt_pulse_data, sample_pulse], axis=0)

            # print("CHECK input data shape: ", np.shape(input_data))
            # print("CHECK ppg data shape: ", np.shape(gt_ppg_data))
            # print("CHECK spo2 data shape: ", np.shape(gt_spo2_data))

        absorption_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            absorption_list.append(temp_list)

        absorption_list = np.array(absorption_list, dtype=np.float32)
        gt_ppg_data = np.array(gt_ppg_data, dtype=np.float32)
        gt_spo2_data = np.array(gt_spo2_data, dtype=np.float32)
        gt_pulse_data = np.array(gt_pulse_data, dtype=np.float32)

        return input_data, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list


class ViatalSignDataset_ppg_lstm_useregression(data.Dataset):
    def __init__(self, mode='train', cl='', use_gpu = False, seq_len = 100):
        self.mode = mode
        self.cl = cl
        self.use_gpu = use_gpu
        self.seq_len = seq_len

        if mode == "train":
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='train')
        else:
            reflect_list, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list = self.read_vitalsign_dataset(name='test')

        mel_feature_model = VitalSign_Feature_mel_thickness().to('cuda')
        thickness_feature_model = VitalSign_Feature_mel_thickness().to('cuda')

        mel_classifier_model = Classifier(cl_mode="mel").to('cuda')
        thickness_classifier_model = Classifier(cl_mode="thickness").to('cuda')

        mel_regression_model = Regression(cl_mode="mel").to('cuda')
        thickness_regression_model = Regression(cl_mode="thickness").to('cuda')

        path = os.path.dirname(__file__)

        mel_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')
        thickness_feature_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/feature_weight_data')

        mel_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')
        thickness_classify_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/classification_weight_data')

        mel_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_mel_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')
        thickness_regression_path = os.path.join(path, './result/Classify_Weight/vitalsign_thickness_0104_prob_005_input14_m1_epoch5000_addinput3/regression_weight_data')

        mel_feature_model.load_state_dict(torch.load(mel_feature_path))
        mel_classifier_model.load_state_dict(torch.load(mel_classify_path))
        mel_regression_model.load_state_dict(torch.load(mel_regression_path))

        thickness_feature_model.load_state_dict(torch.load(thickness_feature_path))
        thickness_classifier_model.load_state_dict(torch.load(thickness_classify_path))
        thickness_regression_model.load_state_dict(torch.load(thickness_regression_path))

        mv_window = 30

        mf_absorption_list= []
        for i in range(len(absorption_list)):
            if i < mv_window:
                mf_absorption_list.append(absorption_list[i])
            else:
                mf_absorption_list.append(list(np.average(absorption_list[i - mv_window:i, :], axis=0)))

        # mel_x = mel_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        mel_x = mel_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        mel_out = mel_classifier_model(mel_x)
        mel_prob = F.softmax(mel_out, dim=1)

        mel_value = mel_regression_model(mel_prob)

        mel_value = mel_value.detach().cpu().numpy()
        mel_prob = mel_prob.detach().cpu().numpy()

        # thickness_x = thickness_feature_model(torch.FloatTensor(reflect_list).to('cuda'))
        thickness_x = thickness_feature_model(torch.FloatTensor(mf_absorption_list).to('cuda'))
        thickenss_out = thickness_classifier_model(thickness_x)
        thickness_prob = F.softmax(thickenss_out, dim=1)
        thickness_value = thickness_regression_model(thickness_prob)

        thickness_value = thickness_value.detach().cpu().numpy()
        thickness_prob = thickness_prob.detach().cpu().numpy()

        absorption_list1 = np.concatenate((absorption_list, mel_prob, thickness_prob), axis=1)
        absorption_list2 = np.concatenate((absorption_list, mel_value, thickness_value), axis=1)
        reflect_list_ = np.concatenate((reflect_list, mel_value, thickness_value), axis=1)
        # reflect_list = np.concatenate((mf_reflect_list, mel_prob, thickness_prob), axis=1)

        print("CHECK absorption shape 11: ", np.shape(absorption_list))

        sequence_absorption = []
        sequence_absorption2 = []
        sequence_reflectance = []
        sequence_ppg = []
        sequence_spo2 = []
        sequence_pulse = []
        file_start = 0
        file_check_idx = 0

        # gt_ppg_data = np.delete(gt_ppg_data, 0)
        # gt_spo2_data = np.delete(gt_spo2_data, 0)

        for seq_i in range(1, len(absorption_list1), 5):
            if seq_i >= sample_time_list[file_check_idx]:
                file_start = sample_time_list[file_check_idx]
                file_check_idx = file_check_idx+1
                # gt_ppg_data = np.delete(gt_ppg_data, seq_i)
                # gt_spo2_data = np.delete(gt_spo2_data, seq_i)
            # elif (seq_i - file_start) > 3600:
            #     file_start = seq_i

            if (seq_i - file_start) < self.seq_len:
                continue
            else:
                # if self.use_gpu == True:
                #     sequence_absorption.append(torch.FloatTensor(absorption_list[file_start:seq_i+1]).to('cuda'))
                #     sequence_reflectance.append(torch.FloatTensor(reflect_list[file_start:seq_i+1]).to('cuda'))
                #     sequence_ppg.append(torch.FloatTensor(gt_ppg_data[file_start:seq_i+1]).to('cuda'))
                #     sequence_spo2.append(torch.FloatTensor(gt_spo2_data[file_start:seq_i+1]).to('cuda'))
                # else:
                #     sequence_absorption.append(torch.FloatTensor(absorption_list[file_start:seq_i+1]))
                #     sequence_reflectance.append(torch.FloatTensor(reflect_list[file_start:seq_i+1]))
                #     sequence_ppg.append(torch.FloatTensor(gt_ppg_data[file_start:seq_i+1]))
                #     sequence_spo2.append(torch.FloatTensor(gt_spo2_data[file_start:seq_i+1]))

                if self.use_gpu == True:
                    sequence_absorption.append(absorption_list1[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption2.append(absorption_list2[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i-(self.seq_len-1):seq_i+1])

                    # print("CHECK sequence_absorption : ", np.shape(sequence_absorption))
                    #
                    # plt.figure()
                    # plt.plot(sequence_ppg[-1]/200)
                    # plt.plot((np.array(sequence_reflectance)[-1, :, 6]*100)-23)
                    # plt.show()


                else:
                    sequence_absorption.append(absorption_list1[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_absorption2.append(absorption_list2[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_reflectance.append(reflect_list[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_ppg.append(gt_ppg_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_spo2.append(gt_spo2_data[seq_i-(self.seq_len-1):seq_i+1])
                    sequence_pulse.append(gt_pulse_data[seq_i - (self.seq_len - 1):seq_i + 1])

                # print("CHECK sequence_absorption shape : ", np.shape(sequence_absorption))

        # self.sequence_absorption = sequence_absorption
        # self.sequence_reflectance = sequence_reflectance
        # self.sequence_ppg = sequence_ppg
        # self.sequence_spo2 = sequence_spo2

        print("CHECK absorption shape22: ", len(sequence_absorption))

        if self.use_gpu == True:
            self.absorption_list = torch.FloatTensor(absorption_list1).to('cuda')
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data).to('cuda')
            self.gt_spo2_data = torch.FloatTensor(gt_spo2_data).to('cuda')

            self.sequence_absorption = torch.FloatTensor(sequence_absorption).to('cuda')
            self.sequence_absorption2 = torch.FloatTensor(sequence_absorption2).to('cuda')
            self.sequence_reflectance = torch.FloatTensor(sequence_reflectance).to('cuda')
            self.sequence_ppg = torch.FloatTensor(sequence_ppg).to('cuda')
            self.sequence_spo2 = torch.FloatTensor(sequence_spo2).to('cuda')
            self.sequence_pulse = torch.FloatTensor(sequence_pulse).to('cuda')
        else:
            self.absorption_list = torch.FloatTensor(absorption_list1)
            self.gt_ppg_data = torch.FloatTensor(gt_ppg_data)
            self.gt_spo2_data = torch.Tensor(gt_spo2_data)

            self.sequence_absorption = torch.Tensor(sequence_absorption)
            self.sequence_absorption2 = torch.Tensor(sequence_absorption2)
            self.sequence_reflectance = torch.Tensor(sequence_reflectance)
            self.sequence_ppg = torch.Tensor(sequence_ppg)
            self.sequence_spo2 = torch.Tensor(sequence_spo2)
            self.sequence_pulse = torch.Tensor(sequence_pulse)

    def __getitem__(self, index):
        # abs = self.absorption_list[index]
        sequence_absorption = self.sequence_absorption[index]
        sequence_absorption2 = self.sequence_absorption2[index]
        sequence_reflectance = self.sequence_reflectance[index]
        ppg_data = self.sequence_ppg[index]
        spo2_data = self.sequence_spo2[index]
        pulse_data = self.sequence_pulse[index]

        # sequence_absorption = torch.FloatTensor(sequence_absorption)
        # sequence_reflectance = torch.FloatTensor(sequence_reflectance)
        # ppg_data = torch.FloatTensor(ppg_data)
        # spo2_data = torch.FloatTensor(spo2_data)

        return sequence_absorption, sequence_absorption2, sequence_reflectance, ppg_data, spo2_data, pulse_data

    def __len__(self):
        return len(self.sequence_absorption)

    def read_vitalsign_dataset(self, name):
        '''
        Read Data
        '''
        path = os.path.dirname(__file__)

        fileNameList = os.listdir(path + '/check_data/total_data1/')

        input_data = []
        gt_ppg_data = []
        gt_spo2_data = []

        sample_time_list = []

        for fn in fileNameList:
            # temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            temp_raw_data = np.load(path + '/check_data/total_data1/{}/re_forehead_human.npy'.format(fn), allow_pickle=True)
            measure_time = read_measurement_elapsed_time(fn)
            gt_ppg, gt_ppg_time = read_ppg_data(fn)
            gt_spo2, gt_spo2_time = read_spo2_data(fn)
            gt_pulse, gt_pulse_time = read_pulse_data(fn)

            roi_data = []

            for i in range(len(temp_raw_data)):
                temp_data = np.average(np.reshape(temp_raw_data[i][:14, :, :], (14, -1)), axis=1)
                roi_data.append(temp_data)

            roi_data = np.array(roi_data)

            if name == 'train':
                time_start_idx = 0
                time_end_idx = 120
            else:
                time_start_idx = 100
                time_end_idx = 160

            start_idx = int(np.where(time_start_idx-1 <= measure_time)[0][0])
            end_idx = int(np.where(measure_time <= time_end_idx+1)[0][-1])

            temp_interpolation = []
            for temp_wave in range(14):
                f_linear_ref = interpolate.interp1d(measure_time[start_idx:end_idx], roi_data[start_idx:end_idx, temp_wave], kind='linear')
                sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
                sample_ref = f_linear_ref(sample_tiem)
                sample_ref = sample_ref[:, np.newaxis]

                if len(temp_interpolation) == 0:
                    temp_interpolation = sample_ref
                else:
                    temp_interpolation = np.concatenate([temp_interpolation, sample_ref], axis=1)

            sample_tiem = np.arange(time_start_idx, time_end_idx, (1 / 60))
            f_linear_ppg = interpolate.interp1d(gt_ppg_time, gt_ppg, kind='linear')
            sample_ppg = f_linear_ppg(sample_tiem)

            f_linear_spo2 = interpolate.interp1d(gt_spo2_time, gt_spo2, kind='linear')
            sample_spo2 = f_linear_spo2(sample_tiem)

            f_linear_pulse = interpolate.interp1d(gt_pulse_time, gt_pulse, kind='linear')
            sample_pulse = f_linear_pulse(sample_tiem)

            if len(sample_time_list) == 0:
                sample_time_list.append(len(sample_tiem))
            else:
                sample_time_list.append(sample_time_list[-1]+len(sample_tiem))

            if len(input_data) == 0 :
                input_data = temp_interpolation
                gt_ppg_data = sample_ppg
                gt_spo2_data = sample_spo2
                gt_pulse_data = sample_pulse
            else:
                input_data = np.concatenate([input_data, temp_interpolation], axis=0)
                gt_ppg_data = np.concatenate([gt_ppg_data, sample_ppg], axis=0)
                gt_spo2_data = np.concatenate([gt_spo2_data, sample_spo2], axis=0)
                gt_pulse_data = np.concatenate([gt_pulse_data, sample_pulse], axis=0)

            # print("CHECK input data shape: ", np.shape(input_data))
            # print("CHECK ppg data shape: ", np.shape(gt_ppg_data))
            # print("CHECK spo2 data shape: ", np.shape(gt_spo2_data))

        absorption_list = []

        k_list = []
        for i in range(len(input_data)):
            shading, k = calculate_k(-(np.log(input_data[i][0])),
                                     -(np.log(input_data[i][6])),
                                     -(np.log(input_data[i][13])))

            if i < 30:
                k_list.append(k)

            temp_list = []
            for ii in range(14):  # 25
                temp_list.append(-(np.log(input_data[i][ii])) - k)

            absorption_list.append(temp_list)

        absorption_list = np.array(absorption_list, dtype=np.float32)
        gt_ppg_data = np.array(gt_ppg_data, dtype=np.float32)
        gt_spo2_data = np.array(gt_spo2_data, dtype=np.float32)
        gt_pulse_data = np.array(gt_pulse_data, dtype=np.float32)

        return input_data, absorption_list, gt_ppg_data, gt_spo2_data, gt_pulse_data, sample_time_list




