import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset1 import ViatalSignDataset_ppg_lstm2

from torch.autograd import Variable

""" 정상상태로 3분씩 수집한 Dataset1을 이용하여, 
    특정 실험자의 data를 Test data로 사용하고 나머지 실험자의 데이터를 모두 Training data로 사용 (Cross validation 방식)
"""

class VitalSign_Spo2(nn.Module):
    def __init__(self, feature_size=25, hidden_size=30, seq_len = 300):
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

    # Model의 Sequence Length와 Hidden Size 설정
    seq_len = 100
    hidden_size = 30

    # ROI 설정
    roi = 'forehead'
    # roi = 'ueye'
    # roi = 'cheek'
    # roi = 'unose'

    # 멜라닌과 피부두께 확률분포를 입력으로 함께 사용하는 경우 True로 설정, Absorbance만 사용하는 경우 False로 설정
    use_mel_thick = True

    # Teat data로 사용할 File 이름
    test_name_list = ['1_ar', 'aron']

    # Model Weight를 저장할 파일 경로 및 이름
    save_dir = "test _vitalsign_0409_predict_spo2_lstm_l2_dropno_dataset1_{}_seq{}_hidden{}_crossvali".format(roi, seq_len,
                                                                                                 hidden_size)

    path = os.path.dirname(__file__)

    if os.path.isdir("result/{}".format(save_dir)) == False:
        os.mkdir("result/{}".format(save_dir))

    for tn in test_name_list:
        lstm_path1 = os.path.join(path, './result/{}/weight_data1_{}'.format(save_dir, tn))
        lstm_path2 = os.path.join(path, './result/{}/weight_data2_{}'.format(save_dir, tn))


        ############# 1. Train Feature Model ###################
        print("**************************************************")
        print("************  Train Model  {} *************".format(tn))
        print("**************************************************")

        # Spo2 Model 생성
        if use_gpu == True:
            if use_mel_thick == True:
                spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len= seq_len).to('cuda')
            else:
                spo2_model = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len= seq_len).to('cuda')

        else:
            if use_mel_thick == True:
                spo2_model = VitalSign_Spo2(feature_size=25, hidden_size=hidden_size, seq_len= seq_len)
            else:
                spo2_model = VitalSign_Spo2(feature_size=14, hidden_size=hidden_size, seq_len= seq_len)

        # Dataloader 선언
        training_dataset = ViatalSignDataset_ppg_lstm2(mode='train', use_gpu = True, seq_len=seq_len, roi = roi, test_name = tn)
        data_loader = DataLoader(training_dataset, batch_size=3000, shuffle=False)

        test_dataset = ViatalSignDataset_ppg_lstm2(mode='test', use_gpu = True, seq_len=seq_len, roi=roi, test_name = tn)
        test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # 학습을 위한 Optimizer 선언
        optimizer = optim.Adam(spo2_model.parameters(), lr=0.001)

        # Loss Function
        criterion = nn.MSELoss()

        best_loss = 700
        best_test_loss = 700
        epochs = 1000 # 3000

        train_loss_save = []
        test_loss_save = []
        test_loss_save2 = []

        for epoch in range(epochs):
            if epoch == 300:
                optimizer = optim.Adam(spo2_model.parameters(), lr=0.001)
            if epoch == 1000:
                optimizer = optim.Adam(spo2_model.parameters(), lr=0.0005)
            if epoch == 2000:
                optimizer = optim.Adam(spo2_model.parameters(), lr=0.0003)
            if epoch == 3000:
                optimizer = optim.Adam(spo2_model.parameters(), lr=0.0001)
            if epoch == 4000:
                optimizer = optim.Adam(spo2_model.parameters(), lr=0.00005)

            #
            running_loss = []
            running_test_loss = []
            for input_data, input_concat_data, input_ref, ppg_data, spo2_data, _ in data_loader:
                spo2_model.train()

                if use_mel_thick == True:
                    pred_spo2 = spo2_model(input_concat_data)
                else:
                    pred_spo2 = spo2_model(input_data)

                pred_spo2 = torch.squeeze(pred_spo2)
                ppg_data = torch.squeeze(ppg_data)
                spo2_data = spo2_data[:, -1]

                loss = criterion(pred_spo2, spo2_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.append(loss.detach().cpu().numpy())

            # 10 Epoch 단위로 Test data에 대한 Loss를 확인하고 Model을 Save함.
            if epoch % 10 == 0:
                with torch.no_grad():
                    mean_loss = np.mean(running_loss)

                    spo2_model.eval()

                    for input_data_t, input_concat_data_t, input_ref_t, ppg_data_t, spo2_data_t, _ in test_data_loader:
                        if use_mel_thick == True:
                            pred_spo2_t = spo2_model(input_concat_data_t)
                        else:
                            pred_spo2_t = spo2_model(input_data_t)

                        pred_spo2_t = torch.squeeze(pred_spo2_t)
                        ppg_data_t = torch.squeeze(ppg_data_t)
                        spo2_data_t = spo2_data_t[:, -1]

                        test_loss = criterion(pred_spo2_t, spo2_data_t)

                        running_test_loss.append(test_loss.detach().cpu().numpy())

                    mean_test_loss = np.mean(running_test_loss)

                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        print("Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f} (Save Model)".format(epoch + 1, epochs, mean_loss, mean_test_loss))
                        torch.save(spo2_model.state_dict(), lstm_path1)

                    else:
                        print("Epoch: {}/{} - Loss: {:.4f} , Test Loss: {:.4f}".format(epoch + 1, epochs, mean_loss, mean_test_loss))

                    if mean_test_loss < best_test_loss and mean_test_loss > 0.5:
                        best_test_loss = mean_test_loss
                        torch.save(spo2_model.state_dict(), lstm_path2)
                        print("[Save Feature Netwrok 2]")
