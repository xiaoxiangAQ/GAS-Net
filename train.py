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

def train(input_train, input_val, model_dir, load_model, network_name, batch_size, arg_lab, 
    EPOCH=100, LR=0.0001):

    if isinstance(input_train, str):
        data_loader_train_img = dataprogress(input_train, batch_size)
    else:
        data_loader_train_img = torch.utils.data.DataLoader(dataset=input_train,
            batch_size = batch_size, shuffle = True, num_workers=4)
        data_loader_test_img = torch.utils.data.DataLoader(dataset=input_val,
            batch_size = 1, shuffle = True, num_workers=4)
    model = network_name().to(device, dtype=torch.float)

    # set optimization
    optimizer = optim.Adam(itertools.chain(model.parameters()), lr= LR, betas=(0.9, 0.999))

    if load_model:
        checkpoint = torch.load(load_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func2 = nn.BCELoss().cuda()
    if arg_lab:
        model_name = '_lab_model.pkl'
    else:
        model_name = '_model.pkl'


    # training
    for epoch in range(start_epoch+1, EPOCH+1):
        train_bar = tqdm(data_loader_train_img)
        running_results = {'batch_sizes': 0, 'SR_loss':0, 'CD_loss':0, 'loss': 0 }
        model.train()
        for data1, data2, labels in train_bar:
            running_results['batch_sizes'] += batch_size

            data1 = data1.to(device, dtype=torch.float).cuda()
            data2 = data2.to(device, dtype=torch.float).cuda()
            labels = labels.to(device, dtype=torch.float)
            labels = torch.argmax(labels, 1).unsqueeze(1).float().cuda()
            
            dist = model(data1,data2)
            # calculate IoU
            CD_loss = loss_func2(dist, labels)

            model.zero_grad()
            CD_loss.backward()
            optimizer.step()

            # loss for current batch before optimization
            running_results['CD_loss'] += CD_loss.item() * batch_size

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, EPOCH,
                    running_results['CD_loss'] / running_results['batch_sizes']))


        # model saving and val set evaluation
        if epoch% 5 == 0:
            output_name = model_dir+str(epoch)+model_name
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, output_name)

            model.eval()
            test_bar = tqdm(data_loader_test_img)
            inter, unin = 0,0
            matrix_all=np.array([0,0,0,0])
            valing_results = {'loss':0,'SR_loss': 0, 'CD_loss':0, 'batch_sizes': 0, 'IoU': 0}
            for img1, img2, labels_test in test_bar:
                valing_results['batch_sizes'] += 1
                        
                img1 = img1.to(device, dtype=torch.float)
                img2 = img2.to(device, dtype=torch.float)
                labels_test = labels_test.to(device, dtype=torch.float)
                labels_test = torch.argmax(labels_test, 1).unsqueeze(1).float()

                dist = model(img1,img2)
                # calculate IoU
                gt_value = (labels_test > 0).float()
                prob = (dist > 0.5).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                # loss for current batch before optimization
                valing_results['IoU'] = (inter * 1.0 / unin)

                test_bar.set_description(
                    desc='IoU: %.4f' % ( valing_results['IoU'],
                    ))

                # evaluation
                matrix = Statistics(gt_value, result)
                matrix_all = matrix+matrix_all

            acc, recall, precision, f1, K = Calculation(matrix_all)

            print("OA:",acc,"P值:",precision,"R值:",recall,"F1值:",f1,"kappa值:",K)
            del img1, img2, labels_test
        del data1, data2, labels


if __name__ == '__main__':

    ############### training data path and model dir
    data_A_dir = './data/levir_CD' 
    data_train_dir = data_A_dir+'/train/images' 
    data_val_dir = data_A_dir+'/val/images' 
    data_test_dir = data_A_dir+'/test/images' 
    network_name = CDNet_L
    batch_size = 16

    ### train&val
    model_dir = './data/levir_CD/resnet18-'
    modelfile = False
    input_train = Dataset_self(data_train_dir) 
    input_val = Dataset_self(data_val_dir) 
    train(input_train, input_val, model_dir, modelfile, network_name, batch_size, arg_lab=False, 
        EPOCH=100, LR=0.0002)

    ### test levir_CD
    modelfile = './data/levir_CD/models/100_model.pkl'
    input_test = Dataset_self(data_test_dir)
    test(input_test, modelfile, network_name, BATCH_SIZE=4)

