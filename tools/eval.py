# 使得批量化精度评定
import  os
from tqdm import tqdm
from libtiff import TIFF
import cv2
import numpy as np

def Statistics(y_target,y_pred):

    # y_pred = y_pred/255
    # print(y_pred)
    # y_target = y_target/255
    # print(y_target)
    y_ones = np.ones_like(y_pred)
    y_zeros = np.zeros_like(y_pred)

    TP = np.count_nonzero(y_pred*y_target == y_ones)
    FP = float(np.count_nonzero(y_pred))-TP
    FN = float(np.count_nonzero(y_target))-TP
    TN = np.count_nonzero(y_pred+y_target == y_zeros)

    # #经过验证：TP + TN + FP + FN = y_target.size
    # print("TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN,"4:",TP + TN + FP + FN)
            # print('TP + TN + FP + FN',TP + TN + FP + FN)
    metric = np.array([TP, FP, TN, FN])
    return metric

def Calculation(metric):

    [TP, FP, TN, FN] = metric
    acc = (TP + TN) / (TP + TN + FP + FN)  # 精确率
    recall = (TP) / (TP + FN)  # 召回率
    precision = TP / (TP + FP)  # 精确lv
    f1 = 2 * TP / (2 * TP + FP + FN)
    N = TP + TN + FP +FN
    po = (TP + TN) / N
    Pe = ((TP + FP ) * (TP + FN) + (TN + FP) * (TN + FN )) / (N * N)
    KAPPA = (po - Pe) / ( 1 - Pe)
    return acc, recall, precision, f1, KAPPA


def accuracy(input, target):
    return 100 * float(torch.count_nonzero(input == target)) / target.size

def Show_All_Results(input, target):
    y_target = []
    y_pred = []
    target_fns  = os.listdir(target)
    target_fns.sort()
    for ims in tqdm(target_fns):
        target = TIFF.open(os.path.join(target, ims),mode='r')
        target = target.read_image()
        # target = cv2.imread(os.path.join(target,ims),cv2.IMREAD_GRAYSCALE)
        # pred = cv2.imread(os.path.join(input, ims), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(input, os.path.basename(ims).replace('.tif','.png')),cv2.IMREAD_GRAYSCALE)
        
        _, y_target = cv2.threshold(target, 1, 255, cv2.THRESH_BINARY)
        _, y_pred = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

        # for i in range(256):
        #     for j in range(256):
        #         if target[i][j] / 255 > 0:
        #             target[i][j] = 255
        #         if target[i][j] / 255 == 0:
        #             target[i][j] = 0
        #         if pred[i][j] / 255 > 0:
        #             pred[i][j] = 255
        #         if pred[i][j] / 255 == 0:
        #             pred[i][j] = 0
        #         y_target.append((target[i][j]))
        #         y_pred.append((pred[i][j]))
    acc, recall, precision,f1,K = Calculation(y_target,y_pred)
    print("OA:",acc,"P值：",precision,"R值：",recall,"F1值:",f1,"kappa值：",K)
    return acc,recall,precision,f1,K

if __name__ =='__main__':
    label_folder = './data/levir_CD/Label'
    pred_folder  = './data/levir_CD/Label'
    Show_All_Results(pred_folder, label_folder)
