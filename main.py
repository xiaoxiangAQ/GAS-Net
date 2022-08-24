import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
# import torchvision
from data_loader_new import Dataset_self
from data_utils import calMetric_iou
import numpy as np
from model.CDNet_L import CDNet_L
from tools.eval import Statistics,Calculation
import itertools
from tqdm import tqdm
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(input_a, modelfile, network_name, BATCH_SIZE=1):
    
    data_loader_test_img = torch.utils.data.DataLoader(dataset=input_a,
        batch_size = 1,shuffle = True)

    model = network_name()
    # checkpoint = torch.load(load_model)
    #     model.load_state_dict(checkpoint['model'])
        
    model.load_state_dict(torch.load(modelfile)['model'])
    if torch.cuda.is_available():
        model = model.cuda() 
        # model2 = model2.cuda() 
    i_count=1
    matrix_all=np.array([0,0,0,0])

    model.eval()
    test_bar = tqdm(data_loader_test_img)
    inter, unin = 0,0
    matrix_all=np.array([0,0,0,0])
    valing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0}
    for img1, img2, labels in test_bar:
        valing_results['batch_sizes'] += 1

        data1 = img1.to(device, dtype=torch.float)
        data2 = img2.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        labels = torch.argmax(labels, 1).unsqueeze(1).float()

        gt_value = labels.float()
        dist = model(data1,data2)

        prob = (dist > 0.5).float()
        prob = prob.cpu().detach().numpy()

        gt_value = gt_value.cpu().detach().numpy()
        gt_value = np.squeeze(gt_value)
        result = np.squeeze(prob)
        # print(result.shape,'result.shape')
        intr, unn = calMetric_iou(gt_value, result)
        inter = inter + intr
        unin = unin + unn

        # loss for current batch before optimization
        valing_results['IoU'] = (inter * 1.0 / unin)

        test_bar.set_description(
            desc='IoU: %.4f' % ( valing_results['IoU'],
            ))

        ###evaluation
        matrix = Statistics(gt_value, result)
        matrix_all = matrix+matrix_all

    acc, recall, precision, f1, K = Calculation(matrix_all)
    print("OA:",acc,"P值:",precision,"R值:",recall,"F1值:",f1,"kappa值:",K)


if __name__ == '__main__':

    ############### training data path and model dir
    data_A_dir = './data/levir_CD' 
    data_test_dir = data_A_dir+'/test/images' 
    network_name = CDNet_L
    batch_size = 16

    ### test levir_CD
    modelfile = './data/levir_CD/models/100_model.pkl'
    input_test = Dataset_self(data_test_dir)
    test(input_test, modelfile, network_name, BATCH_SIZE=4)


  #   ### test SVCD
  #   modelfile = './data/SVCD/leveb/models/100_model.pkl'
  #   input_test = Dataset_self(data_test_dir)
  #   test(input_test, modelfile, network_name, BATCH_SIZE=4)

