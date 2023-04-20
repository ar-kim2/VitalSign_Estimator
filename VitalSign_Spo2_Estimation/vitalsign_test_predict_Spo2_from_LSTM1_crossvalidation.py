import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset1 import ViatalSignDataset_ppg_lstm
from dataset1 import ViatalSignDataset_ppg_lstm2

from torch.autograd import Variable

class VitalSign_Spo2(nn.Module):
    def __init__(self, feature_size=14, hidden_size=30, seq_len = 300):
        super(VitalSign_Spo2, self).__init__()

        self.feature_size = feature_size # FFT Result Size
        self.hidden_size = hidden_size # output pulse value = 1
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
    # save_dir = "vitalsign_0409_predict_spo2_lstm_l2_dropno_dataset1_{}_seq{}_hidden{}_nomelthick_crossvali".format(roi, seq_len,
    #                                                                                                     hidden_size)

    save_dir = "vitalsign_0409_predict_spo2_lstm_l2_dropno_dataset1_{}_seq{}_hidden{}_crossvali".format(roi, seq_len,
                                                                                                        hidden_size)
    path = os.path.dirname(__file__)

    use_mel_thick = True

    test_name_list = ['1_ar', 'aron']

    for tn in test_name_list:
        lstm_path2 = os.path.join(path, './result/{}/weight_data2_{}'.format(save_dir, tn))

        if use_gpu == True:
            if use_mel_thick == True:
                spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len=seq_len).to('cuda')
            else:
                spo2_model = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len=seq_len).to('cuda')
        else:
            if use_mel_thick == True:
                spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len=seq_len)
            else:
                spo2_model = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len=seq_len)

        spo2_model.load_state_dict(torch.load(lstm_path2))

        dataset = ViatalSignDataset_ppg_lstm2(mode='test', use_gpu = True, seq_len=seq_len, roi=roi, test_name = tn)
        test_data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        criterion = nn.MSELoss()
        running_test_loss = []

        with torch.no_grad():
            spo2_model.eval()

            for input_data_t, input_data_concat_t, input_ref_t, ppg_data_t, spo2_data_t, _ in test_data_loader:
                if use_mel_thick == True:
                    pred_spo2_t = spo2_model(input_data_concat_t)
                else:
                    pred_spo2_t = spo2_model(input_data_t)

                pred_spo2_t = torch.squeeze(pred_spo2_t)

                ppg_data_t = torch.squeeze(ppg_data_t)
                spo2_data_t = spo2_data_t[:, -1]

                loss_list = ((pred_spo2_t - spo2_data_t) ** 2) ** (1 / 2)

                test_loss = criterion(pred_spo2_t, spo2_data_t)

                running_test_loss.append(test_loss.detach().cpu().numpy())

            pred_spo2_t = pred_spo2_t.detach().cpu().numpy()
            spo2_data_t = spo2_data_t.detach().cpu().numpy()

            mean_test_loss = np.mean(running_test_loss)

            print("Test {} RMSE : {}".format(tn, mean_test_loss))

            # loss_list = ((pred_spo2_t - spo2_data_t) ** 2) ** (1 / 2)
            loss_mean = test_loss.item()

            deviation_list = []

            for di in range(len(loss_list)):
                deviation_list.append(((loss_list[di] - loss_mean) ** 2).item())

            sto_var = np.mean(deviation_list)

            print("Test {} Standard variation mean : {}".format(tn, np.sqrt(sto_var)))

            # plt.figure()
            # plt.plot(pred_spo2_t, label='Predicted Value')
            # plt.plot(spo2_data_t,c = 'orange', label='Ground Truth')
            # plt.xlabel('Test data')
            # plt.ylabel('Spo2 (%)')
            # plt.legend()
            # plt.show()