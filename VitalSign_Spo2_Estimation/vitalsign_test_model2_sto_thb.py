import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset2 import ViatalSignDataset_triplet
from dataset2 import ViatalSignDataset_class
from dataset2 import ViatalSignDataset_regression

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


class VitalSign_Feature(nn.Module):
    def __init__(self):
        super(VitalSign_Feature, self).__init__()

        self.input_dim = 25 #36 #43

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
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_dim = 128

        self.layer11 = nn.Linear(self.input_dim, 128)
        self.layer12 = nn.Linear(128, 128)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128, 128)
        self.layer15 = nn.Linear(128, 130)
        # self.layer15 = nn.Linear(128, 28)
        # self.layer15 = nn.Linear(128, 77)
        #self.layer15 = nn.Linear(128, 42)

    def forward(self, x):
        x1 = F.leaky_relu(self.layer11(x))
        x1 = F.leaky_relu(self.layer12(x1))
        x1 = F.leaky_relu(self.layer13(x1))
        x1 = F.leaky_relu(self.layer14(x1))
        x1 = self.layer15(x1)

        return x1

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.input_dim = 130 # 77 #28 #49

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


if __name__ == '__main__':
    use_gpu = True
    class_mode = ""

    # save_dir = "vitalsign_0224_ppg_spo2"
    save_dir = "vitalsign_0224_ppg_spo2_undereye"

    path = os.path.dirname(__file__)

    feature_path = os.path.join(path, './result/{}/feature_weight_data'.format(save_dir))
    feature_path2 = os.path.join(path, './result/{}/feature_weight_data2'.format(save_dir))

    classify_path = os.path.join(path, './result/{}/classification_weight_data'.format(save_dir))
    classify_path2 = os.path.join(path, './result/{}/classification_weight_data2'.format(save_dir))

    regression_path_sto = os.path.join(path, './result/{}/regression_sto_weight_data'.format(save_dir))
    regression_path_sto2 = os.path.join(path, './result/{}/regression_sto_weight_data2'.format(save_dir))

    regression_path_thb = os.path.join(path, './result/{}/regression_ppg_weight_data'.format(save_dir))
    regression_path_thb2 = os.path.join(path, './result/{}/regression_ppg_weight_data2'.format(save_dir))

    if os.path.isdir("result/{}".format(save_dir)) == False:
        os.mkdir("result/{}".format(save_dir))


    ############# 2. Train Classification Model ###################
    print("**************************************************")
    print("****** 2.  Test Classification Model ************")
    print("**************************************************")

    if use_gpu == True:
        feature_model = VitalSign_Feature().to('cuda')
    else:
        feature_model = VitalSign_Feature()

    feature_model.load_state_dict(torch.load(feature_path))
    feature_model.eval()

    temper_value = 1

    if use_gpu == True:
        classifier_model = Classifier().to('cuda')
    else:
        classifier_model = Classifier()

    classifier_model.load_state_dict(torch.load(classify_path))

    test_data_len = len(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu))
    test_loader = DataLoader(ViatalSignDataset_class(mode='val', cl=class_mode, use_gpu=use_gpu), batch_size=test_data_len,
                             shuffle=False)

    classifier_model.eval()

    for anchor, total_label3 in test_loader:
        x_data = feature_model(anchor)
        classifier_model.eval()

        pred_prob = classifier_model(x_data)
        pred_prob = pred_prob * temper_value

        pred_prob = F.softmax(pred_prob, dim=1)
        test_loss = torch.mean(-(torch.sum(total_label3 * torch.log(pred_prob+0.000000000001), dim=1)))  # Cross Entropy Loss

        top_p, top_class = pred_prob.topk(1, dim=1)

        total_cnt = 0
        equal_cnt = 0

    print("Classification Cross Entropy Loss : {}".format(test_loss))


    print("**************************************************")
    print("****** 3.  Train Regression Model (Sto) ************")
    print("**************************************************")
    classifier_model.eval()

    if use_gpu == True:
        Reg_model = Regression().to('cuda')
    else:
        Reg_model = Regression()

    Reg_model.load_state_dict(torch.load(regression_path_sto2))

    test_data_len = len(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu))
    test_loader = DataLoader(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu), batch_size=test_data_len, shuffle=False)

    criterion = nn.MSELoss()

    with torch.no_grad():
        for anchor_t, ppg_label_t, st_label_t in test_loader:
            x_data = feature_model(anchor_t)
            pred_prob = classifier_model(x_data)
            # pred_prob = pred_prob * temper_value
            pred_prob = F.softmax(pred_prob, dim=1)

            Reg_model.eval()
            pred_value = Reg_model(pred_prob)
            pred_value = pred_value.squeeze()
            test_loss = criterion(pred_value, st_label_t)
            test_loss = torch.sqrt(test_loss)

            print("Sto Regression RMSE : {}".format(test_loss))

            loss_list = ((pred_value - st_label_t)**2)**(1/2)
            loss_mean = test_loss.item()

            deviation_list = []

            for di in range(len(loss_list)):
                deviation_list.append(((loss_list[di]-loss_mean)**2).item())

            sto_var = np.mean(deviation_list)
            print("Variance mean : ", sto_var)
            print("Standard variation mean : ", np.sqrt(sto_var))

            label_sort = []
            pred_sort = []

            prev_value_result = []
            gt_value_list = []

            for i in range(len(pred_value)):
                prev_value_result.append(pred_value[i].item())
                gt_value_list.append(st_label_t[i].item())

            for i in range(len(gt_value_list)):
                min_idx = np.argmin(gt_value_list)
                label_sort.append(gt_value_list[min_idx])
                pred_sort.append(prev_value_result[min_idx])
                gt_value_list = np.delete(gt_value_list, min_idx)
                prev_value_result = np.delete(prev_value_result, min_idx)

            x_axis = []

            for i in range(np.shape(label_sort)[0]):
                x_axis.append(i)

            # plt.figure()
            # plt.scatter(x_axis, pred_sort, label="pred_value", s=3)
            # plt.scatter(x_axis, label_sort, label='gt', s=3)
            # plt.legend()
            # plt.show()

            pred_value = pred_value.detach().cpu().numpy()
            st_label_t = st_label_t.detach().cpu().numpy()

            plt.figure()
            plt.scatter(x_axis, pred_value, label="pred_value", s=3)
            plt.scatter(x_axis, st_label_t, label='gt', s=3)
            plt.legend()
            plt.show()


    print("**************************************************")
    print("****** 3.  Train Regression Model (PPG) ************")
    print("**************************************************")
    classifier_model.eval()

    if use_gpu == True:
        Reg_model2 = Regression().to('cuda')
    else:
        Reg_model2 = Regression()

    Reg_model2.load_state_dict(torch.load(regression_path_thb2))

    test_data_len = len(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu))
    test_loader = DataLoader(ViatalSignDataset_regression(mode='val', cl=class_mode, use_gpu=use_gpu),
                             batch_size=test_data_len, shuffle=False)

    criterion = nn.MSELoss()

    with torch.no_grad():
        for anchor_t, ppg_label_t, st_label_t in test_loader:
            x_data = feature_model(anchor_t)
            pred_prob = classifier_model(x_data)
            # pred_prob = pred_prob * temper_value
            pred_prob = F.softmax(pred_prob, dim=1)

            Reg_model2.eval()
            pred_value = Reg_model2(pred_prob)
            pred_value = pred_value.squeeze()
            test_loss = criterion(pred_value, ppg_label_t)
            test_loss = torch.sqrt(test_loss)

            print("PPG RMSE : {}".format(test_loss))

            loss_list = ((pred_value - ppg_label_t)**2)**(1/2)
            loss_mean = test_loss.item()

            deviation_list = []

            for di in range(len(loss_list)):
                deviation_list.append(((loss_list[di]-loss_mean)**2).item())

            sto_var = np.mean(deviation_list)
            print("Variance mean : ", sto_var)
            print("Standard variation mean : ", np.sqrt(sto_var))

            label_sort = []
            pred_sort = []

            prev_value_result = []
            gt_value_list = []

            for i in range(len(pred_value)):
                prev_value_result.append(pred_value[i].item())
                gt_value_list.append(ppg_label_t[i].item())

            for i in range(len(gt_value_list)):
                min_idx = np.argmin(gt_value_list)
                label_sort.append(gt_value_list[min_idx])
                pred_sort.append(prev_value_result[min_idx])
                gt_value_list = np.delete(gt_value_list, min_idx)
                prev_value_result = np.delete(prev_value_result, min_idx)

            x_axis = []

            for i in range(np.shape(label_sort)[0]):
                x_axis.append(i)

            # plt.figure()
            # plt.scatter(x_axis, pred_sort, label="pred_value", s=3)
            # plt.scatter(x_axis, label_sort, label='gt', s=3)
            # plt.legend()
            # plt.show()

            pred_value = pred_value.detach().cpu().numpy()
            ppg_label_t = ppg_label_t.detach().cpu().numpy()

            plt.figure()
            plt.plot(x_axis, pred_value, label="pred_value")
            plt.scatter(x_axis, pred_value, s=1)
            plt.plot(x_axis, ppg_label_t, label='gt')
            plt.scatter(x_axis, ppg_label_t, s=1)
            plt.legend()
            plt.show()











