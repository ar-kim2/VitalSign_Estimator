import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset1 import ViatalSignDataset_ppg_lstm

from torch.autograd import Variable

""" 정상상태로 3분씩 수집한 Dataset1을 이용하여, 초기 2분은 Training으로, 나머지 1분은 Test로 사용한 모델에 대한 Test Code"""

class VitalSign_Spo2(nn.Module):
    def __init__(self, feature_size=14, hidden_size=30, seq_len = 300):
        super(VitalSign_Spo2, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.layer = 2

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer,
            batch_first=True).cuda()

        self.dense = nn.Linear(self.hidden_size*self.seq_len, 1)

    def forward(self, x):
        hidden = Variable(torch.zeros(self.layer, x.size()[0], self.hidden_size)).to('cuda')
        cell = Variable(torch.zeros(self.layer, x.size()[0], self.hidden_size)).to('cuda')

        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        hidden = outputs.reshape(-1, self.hidden_size * self.seq_len)
        model = self.dense(hidden)

        return model

if __name__ == '__main__':
    use_gpu = True

    # Model의 Sequence Length와 Hidden Size 설정, Trainingd에 설정한 값과 동일하게 설정해야함.
    seq_len = 100
    hidden_size = 30

    # ROI 설정
    roi = 'forehead'
    # roi = 'ueye'
    # roi = 'cheek'
    # roi = 'unose'

    # Load Model
    save_dir = "vitalsign_0408_predict_spo2_lstm_l2_dropno_dataset1_{}_seq{}_hidden{}".format(roi, seq_len, hidden_size)
    save_dir2 = "vitalsign_0408_predict_spo2_lstm_l2_dropno_dataset1_{}_seq{}_hidden{}_nomelthick".format(roi, seq_len, hidden_size)

    path = os.path.dirname(__file__)
    lstm_path1 = os.path.join(path, './result/{}/weight_data2'.format(save_dir))
    lstm_path2 = os.path.join(path, './result/{}/weight_data2'.format(save_dir2))

    if use_gpu == True:
        spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len= seq_len).to('cuda')
        spo2_model_nomelthick = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len= seq_len).to('cuda')
    else:
        spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len= seq_len)
        spo2_model_nomelthick = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len= seq_len)

    spo2_model.load_state_dict(torch.load(lstm_path1))
    spo2_model_nomelthick.load_state_dict(torch.load(lstm_path2))

    # Dataset 설정
    dataset = ViatalSignDataset_ppg_lstm(mode='test', use_gpu = True, seq_len=seq_len, roi=roi)
    test_data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    running_test_loss = []
    running_test_loss2 = []

    with torch.no_grad():
        spo2_model.eval()

        for input_data_t, input_data_concat_t, input_ref_t, ppg_data_t, spo2_data_t, _, mel, thickness in test_data_loader:
            pred_spo2_t = spo2_model(input_data_concat_t)
            pred_spo2_t_no = spo2_model_nomelthick(input_data_t)

            pred_spo2_t = torch.squeeze(pred_spo2_t)
            pred_spo2_t_no = torch.squeeze(pred_spo2_t_no)
            ppg_data_t = torch.squeeze(ppg_data_t)
            spo2_data_t = spo2_data_t[:, -1]

            loss_list = ((pred_spo2_t - spo2_data_t) ** 2) ** (1 / 2)
            loss_list2 = ((pred_spo2_t_no - spo2_data_t) ** 2) ** (1 / 2)

            test_loss = criterion(pred_spo2_t, spo2_data_t)
            test_loss2 = criterion(pred_spo2_t_no, spo2_data_t)

            running_test_loss.append(test_loss.detach().cpu().numpy())
            running_test_loss2.append(test_loss2.detach().cpu().numpy())

        pred_spo2_t = pred_spo2_t.detach().cpu().numpy()
        pred_spo2_t_no = pred_spo2_t_no.detach().cpu().numpy()
        spo2_data_t = spo2_data_t.detach().cpu().numpy()

        print("spo2 pred : ", np.shape(pred_spo2_t))

        # Spo2 예측 결과에 Moving Average를 적용한 결과 확인
        mf_sto_list = []
        for i in range(len(pred_spo2_t)):
            if i < 10:
                mf_sto_list.append(pred_spo2_t[i])
            else:
                mf_sto_list.append(np.average(pred_spo2_t[i - 10:i]))

        print("mf spo2 pred : ", np.shape(mf_sto_list))

        # 예측 결과의 오차율 계산
        mean_test_loss = np.mean(running_test_loss)
        mean_test_loss2 = np.mean(running_test_loss2)

        rmse = np.sqrt(((pred_spo2_t - spo2_data_t) ** 2).mean())

        print("RMSE : ", rmse)

        print("Test MSE : ", mean_test_loss)
        print("Test MSE no mel thick: ", mean_test_loss2)

        # loss_list = ((pred_spo2_t - spo2_data_t) ** 2) ** (1 / 2)
        loss_mean = test_loss.item()

        # 예측 결과의 표준편차 계산
        deviation_list = []
        deviation_list2 = []

        for di in range(len(loss_list)):
            deviation_list.append(((loss_list[di] - loss_mean) ** 2).item())
            deviation_list2.append(((loss_list2[di] - loss_mean) ** 2).item())

        sto_var = np.mean(deviation_list)
        sto_var2 = np.mean(deviation_list2)

        print("Standard variation mean : ", np.sqrt(sto_var))
        print("Standard variation mean no mel thick: ", np.sqrt(sto_var2))

        # 예측 결과 Graph Draw
        time_step = []

        for ti in range(len(pred_spo2_t)):
            time_step.append(ti*(1/10))

        print(np.shape(pred_spo2_t))

        split_idx = 567

        # plt.figure(figsize=(24, 6))
        # plt.subplots_adjust(left=0.1, bottom=0.1,  right=0.9, top=0.9, wspace=0.08, hspace=0.35)
        # plt.subplot(1, 12, 1)
        # plt.plot(time_step[:567], mf_sto_list[:567], label='Predicted Value (use melanin thickness)')
        # plt.plot(time_step[:567], pred_spo2_t_no[:567], c='m', label='Predicted Value')
        # plt.plot(time_step[:567], spo2_data_t[:567], c='orange', label='Ground Truth')
        # plt.ylabel('Spo2 (%)')
        # plt.ylim([94.95, 100.05])
        # plt.yticks(np.arange(95, 100.05, 1))
        # plt.xticks(np.arange(0, 60.05, 30), labels=[])
        #
        # for i in range(2, 13):
        #     plt.subplot(1, 12, i)
        #     mf_sto_list[567*(i-1)] = mf_sto_list[567*(i-1)+5]
        #     mf_sto_list[567*(i-1)+1] = mf_sto_list[567*(i-1)+6]
        #     mf_sto_list[567*(i-1)+2] = mf_sto_list[567*(i-1)+7]
        #     mf_sto_list[567*(i-1)+3] = mf_sto_list[567*(i-1)+8]
        #     mf_sto_list[567*(i-1)+4] = mf_sto_list[567*(i-1)+9]
        #
        #     plt.plot(time_step[:567], mf_sto_list[567*(i-1):567*i], label='Predicted Value (use melanin thickness)')
        #     plt.plot(time_step[:567], pred_spo2_t_no[567*(i-1):567*i], c = 'm', label='Predicted Value')
        #     plt.plot(time_step[:567], spo2_data_t[567*(i-1):567*i], c = 'orange', label='Ground Truth')
        #     plt.ylim([94.95, 100.05])
        #     plt.yticks(np.arange(95, 100.05, 1), labels=[])
        #     plt.xticks(np.arange(0, 60.05, 30), labels=[])
        #
        #     # if i == 6:
        #     #     plt.xlabel('Time (s)')
        #     if i == 12 :
        #         plt.legend()
        # # plt.show()
        #
        # plt.figure(figsize=(24, 6))
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.08, hspace=0.03)
        # plt.subplot(2, 12, 1)
        # plt.plot(time_step[:567], mel[:567] * 100)
        # plt.ylabel('Melanin (%)')
        # plt.ylim([1, 15.8])
        # plt.yticks(np.arange(3, 15.5, 3))
        # plt.xticks(np.arange(0, 60.05, 30), labels=[])
        #
        # for i in range(2, 13):
        #     plt.subplot(2, 12, i)
        #     plt.plot(time_step[:567], mel[567*(i-1):567*i] * 100)
        #     plt.ylim([1, 15.8])
        #     plt.yticks(np.arange(3, 15.5, 3), labels=[])
        #     plt.xticks(np.arange(0, 60.05, 30), labels=[])
        #
        # plt.subplot(2, 12, 13)
        # plt.plot(time_step[:567], thickness[:567]*1000)
        # plt.ylabel('Thickness (um)')
        # plt.ylim([18, 72])
        # plt.yticks(np.arange(20, 70.05, 10))
        # plt.xticks(np.arange(0, 60.05, 30))
        #
        # for i in range(14, 25):
        #     plt.subplot(2, 12, i)
        #     plt.plot(time_step[:567], thickness[567*(i-13):567*(i-12)] * 1000)
        #     plt.ylim([18, 72])
        #     plt.yticks(np.arange(20, 70.05, 10), labels=[])
        #     plt.xticks(np.arange(0, 60.05, 30))
        #
        #     if i == 19:
        #         plt.xlabel('Time (s)')
        # plt.show()


        """""""""""""""""""""""  For presentation    """""""""""""""""""""
        # plt.figure(figsize=(24, 6))
        plt.figure()
        plt.subplots_adjust(left=0.1, bottom=0.1,  right=0.9, top=0.9, wspace=0.08, hspace=0.35)
        plt.subplot(1, 2, 1)
        plt.plot(time_step[:567], mf_sto_list[:567], label='Predicted Value (use melanin thickness)')
        plt.plot(time_step[:567], pred_spo2_t_no[:567], c='m', label='Predicted Value')
        plt.plot(time_step[:567], spo2_data_t[:567], c='orange', label='Ground Truth')
        plt.ylabel('Spo2 (%)')
        plt.ylim([94.95, 100.05])
        plt.yticks(np.arange(95, 100.05, 1))
        plt.xticks(np.arange(0, 60.05, 30), labels=[])

        plt.subplot(1, 2, 2)
        mf_sto_list[567 * (2 - 1)] = mf_sto_list[567 * (2 - 1) + 5]
        mf_sto_list[567 * (2 - 1) + 1] = mf_sto_list[567 * (2 - 1) + 6]
        mf_sto_list[567 * (2 - 1) + 2] = mf_sto_list[567 * (2 - 1) + 7]
        mf_sto_list[567 * (2 - 1) + 3] = mf_sto_list[567 * (2 - 1) + 8]
        mf_sto_list[567 * (2 - 1) + 4] = mf_sto_list[567 * (2 - 1) + 9]

        plt.plot(time_step[:567], mf_sto_list[567 * (2 - 1):567 * 2], label='Predicted Value (use melanin thickness)')
        plt.plot(time_step[:567], pred_spo2_t_no[567 * (2 - 1):567 * 2], c='m', label='Predicted Value')
        plt.plot(time_step[:567], spo2_data_t[567 * (2 - 1):567 * 2], c='orange', label='Ground Truth')
        plt.ylim([94.95, 100.05])
        plt.yticks(np.arange(95, 100.05, 1), labels=[])
        plt.xticks(np.arange(0, 60.05, 30), labels=[])

        plt.figure()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.08, hspace=0.03)
        plt.subplot(2, 2, 1)
        plt.plot(time_step[:567], mel[:567] * 100)
        plt.ylabel('Melanin (%)')
        plt.ylim([1, 15.8])
        plt.yticks(np.arange(3, 15.5, 3))
        plt.xticks(np.arange(0, 60.05, 30), labels=[])

        plt.subplot(2, 2, 2)
        plt.plot(time_step[:567], mel[567 * (2 - 1):567 * 2] * 100)
        plt.ylim([1, 15.8])
        plt.yticks(np.arange(3, 15.5, 3), labels=[])
        plt.xticks(np.arange(0, 60.05, 30), labels=[])

        plt.subplot(2, 2, 3)
        plt.plot(time_step[:567], thickness[:567]*1000)
        plt.ylabel('Thickness (um)')
        plt.ylim([18, 72])
        plt.yticks(np.arange(20, 70.05, 10))
        plt.xticks(np.arange(0, 60.05, 30))

        plt.subplot(2, 2, 4)
        plt.plot(time_step[:567], thickness[567 * (14 - 13):567 * (14 - 12)] * 1000)
        plt.ylim([18, 72])
        plt.yticks(np.arange(20, 70.05, 10), labels=[])
        plt.xticks(np.arange(0, 60.05, 30))

        plt.show()