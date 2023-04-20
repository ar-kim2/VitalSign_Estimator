import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_ppg_lstm


from torch.autograd import Variable

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

    seq_len = 100
    hidden_size = 30

    roi = 'forehead'
    # roi = 'ueye'
    # roi = 'cheek'
    # roi = 'unose'

    save_dir = "vitalsign_0408_predict_spo2_lstm_l2_dropno_dataset2_{}_seq{}_hidden{}_2".format(roi, seq_len, hidden_size)
    # save_dir = "vitalsign_0408_predict_spo2_lstm_l2_dropno_dataset2_{}_seq{}_hidden{}".format(roi, seq_len, hidden_size)
    save_dir2 = "vitalsign_0224_ppg_spo2"

    path = os.path.dirname(__file__)
    lstm_path1 = os.path.join(path, './result/{}/weight_data1'.format(save_dir))
    lstm_path2 = os.path.join(path, './result/{}/weight_data2'.format(save_dir))

    if use_gpu == True:
        spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len=seq_len).to('cuda')
    else:
        spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len=seq_len)

    spo2_model.load_state_dict(torch.load(lstm_path2))

    # test_data_len = len(ViatalSignDataset_ppg_lstm(mode='test', seq_len=seq_len))
    dataset = ViatalSignDataset_ppg_lstm(mode='test', use_gpu = True, seq_len=seq_len, roi=roi)
    # dataset = ViatalSignDataset_ppg_lstm3(mode='test', use_gpu = True, seq_len=seq_len, roi=roi)
    test_data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    criterion = nn.MSELoss()

    train_loss_save = []
    test_loss_save = []
    test_loss_save2 = []

    running_loss = []
    running_test_loss = []

    with torch.no_grad():
        spo2_model.eval()

        for input_data_t, input_data_concat_t, input_ref_t, ppg_data_t, spo2_data_t, _ in test_data_loader:

            pred_spo2_t = spo2_model(input_data_concat_t)

            pred_spo2_t = torch.squeeze(pred_spo2_t)
            ppg_data_t = torch.squeeze(ppg_data_t)
            spo2_data_t = spo2_data_t[:, -1]

            loss_list = ((pred_spo2_t - spo2_data_t) ** 2) ** (1 / 2)

            test_loss = criterion(pred_spo2_t, spo2_data_t)

            running_test_loss.append(test_loss.detach().cpu().numpy())

        pred_spo2_t = pred_spo2_t.detach().cpu().numpy()
        spo2_data_t = spo2_data_t.detach().cpu().numpy()

        path = os.path.dirname(__file__)
        np.save(path+'/result/Estimate_result1/unose_pred_value2.npy', pred_spo2_t)
        # np.save(path+'/result/Estimate_result1/gt_value2.npy', spo2_data_t)

        r_square = 1 - (
                    (np.sum((spo2_data_t - pred_spo2_t) ** 2)) / (np.sum((spo2_data_t - np.mean(spo2_data_t)) ** 2)))

        print("R square : ", r_square)

        mf_sto_list = []
        for i in range(len(pred_spo2_t)):
            if i < 10:
                mf_sto_list.append(pred_spo2_t[i])
            else:
                mf_sto_list.append(np.average(pred_spo2_t[i - 10:i]))

        mean_test_loss = np.mean(running_test_loss)

        print("Test RMSE : ", mean_test_loss)


        loss_mean = test_loss.item()

        deviation_list = []

        for di in range(len(loss_list)):
            deviation_list.append(((loss_list[di] - loss_mean) ** 2).item())

        sto_var = np.mean(deviation_list)
        print("Variance mean : ", sto_var)
        print("Standard variation mean : ", np.sqrt(sto_var))


        time_step = []

        for ti in range(len(pred_spo2_t)):
            time_step.append(ti*(1/10))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # plt.plot(pred_spo2_t, label='Predicted Value')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.35)
        plt.plot(time_step[:2490-600], mf_sto_list[600:2490], label='Predicted Value')
        plt.plot(time_step[:2490-600], spo2_data_t[600:2490], label='Ground Truth')

        plt.plot((time_step[100], time_step[100]), (91, 99.8), c='r')
        plt.plot((time_step[500], time_step[500]), (91, 99.8), c='r')
        plt.plot((time_step[900], time_step[900]), (91, 99.8), c='r')
        plt.plot((time_step[1300], time_step[1300]), (91, 99.8), c='r')

        plt.xlabel('Time (s)')
        plt.ylabel('Spo2 (%)')
        plt.ylim([90.5, 100])
        plt.yticks(np.arange(91, 100, 2))
        plt.legend()
        plt.subplot(1, 2, 2)
        # plt.plot(pred_spo2_t, label='Predicted Value')
        plt.plot(time_step[:len(mf_sto_list)-2491-400], mf_sto_list[2491+400:], label='Predicted Value')
        plt.plot(time_step[:len(mf_sto_list)-2491-400], spo2_data_t[2491+400:], label='Ground Truth')

        plt.plot((time_step[100], time_step[100]), (91, 99.8), c='r')
        plt.plot((time_step[550], time_step[550]), (91, 99.8), c='r')
        plt.plot((time_step[1050], time_step[1050]), (91, 99.8), c='r')
        plt.plot((time_step[1500], time_step[1500]), (91, 99.8), c='r')

        plt.xlabel('Time (s)')
        # plt.ylabel('Spo2 (%)')
        plt.ylim([90.5, 100])
        plt.yticks(np.arange(91, 100, 2))
        plt.legend(loc='upper right')

        plt.show()