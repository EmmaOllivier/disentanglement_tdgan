
import os
import time
import torch
from torch import nn, optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import models_resnet as model
import models_inception
import numpy as np
import pandas as pd
import data_loader
import util
import PIL.Image as Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device="cuda"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_deterministic(seed=0):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def Val_acc(loader, Dis, criterion, device):
    pre_list = []
    GT_list = []
    val_ce = 0
    loop_test = tqdm(loader ,colour='GREEN')
    for i, (batch_val_x, batch_val_y, batch_val_id_y, _) in enumerate(loop_test):
        GT_list = np.hstack((GT_list, batch_val_y.numpy()))
        batch_val_x = Variable(batch_val_x).to(device)
        batch_val_y = Variable(batch_val_y).to(device)
        _, batch_p = Dis(batch_val_x)

        batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
        pre_list = np.hstack((pre_list, batch_result))
      
        val_ce += criterion(batch_p, batch_val_y).cpu().data.numpy()
        val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
        loop_test.set_description(f"Test")
        loop_test.set_postfix(accuracy_pain=val_acc*100)

    val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
    val_ce = val_ce / i

    return val_acc, val_ce


def Val_acc_per_subject(loader, Dis, criterion, device):

    accuracy_list=[]
    pre_list = []
    GT_list = []
    val_ce = 0
    current_subject=-1
    loop_test = tqdm(loader ,colour='GREEN')
    for i, (batch_val_x, batch_val_y, batch_val_id_y) in enumerate(loop_test):
        if (current_subject==batch_val_id_y or current_subject==-1):
            current_subject=batch_val_id_y 
            GT_list = np.hstack((GT_list, batch_val_y.numpy()))
            batch_val_x = Variable(batch_val_x).to(device)
            batch_val_y = Variable(batch_val_y).to(device)
            _, batch_p = Dis(batch_val_x)

            batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
            pre_list = np.hstack((pre_list, batch_result))

            # classification loss
            val_ce += criterion(batch_p, batch_val_y).cpu().data.numpy()
            val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
            loop_test.set_description(f"Test")
            loop_test.set_postfix(accuracy_pain=val_acc*100)
        elif(current_subject!=batch_val_id_y):
            current_subject=batch_val_id_y 
            accuracy_list.append(val_acc)
            pre_list = []
            GT_list = []
            val_ce = 0
            GT_list = np.hstack((GT_list, batch_val_y.numpy()))
            batch_val_x = Variable(batch_val_x).to(device)
            batch_val_y = Variable(batch_val_y).to(device)
            _, batch_p = Dis(batch_val_x)

            batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
            pre_list = np.hstack((pre_list, batch_result))

            val_ce += criterion(batch_p, batch_val_y).cpu().data.numpy()
            val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
            loop_test.set_description(f"Test")
            loop_test.set_postfix(accuracy_pain=val_acc*100)
            val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
            val_ce = val_ce / i

    return accuracy_list


def test():
  
    seed=0
    make_deterministic(seed)
  
    Resolution=128

    pre_root_dir="../model/"
    Biovid_img_all = '../Biovid/'

    tr_test = util.data_adapt(Resolution)

    biovid_annot_test = '../test_set.csv'

    dataset_test = data_loader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)

    expr_tt_loader = torch.utils.data.DataLoader(dataset_test, batch_size=20, shuffle=False, num_workers=4)

    par_Enc_FR_dir = os.path.join(pre_root_dir, 'Enc_FR_G.pkl')
    par_Enc_ER_dir = os.path.join(pre_root_dir, 'Enc_ER_G.pkl')
    par_dec_dir = os.path.join(pre_root_dir, 'dec.pkl')
    par_fc_ER_dir = os.path.join(pre_root_dir, 'fc_ER_G.pkl')

    par_Dis_ER_dir = os.path.join(pre_root_dir, 'Dis_ER.pkl')
    

     
    # load parameters
    print('loading pretrained models......')

    Dis_ER = models_inception.Dis()
    Gen = models_inception.Gen(clsn_ER=2, Nz=256, GRAY=False, Nb=6)
    Dis_ER_val = models_inception.Dis()

    Gen.to(device)
    Dis_ER.to(device)
    Dis_ER_val.to(device)

    Gen.enc_ER.load_state_dict(util.del_extra_keys(par_Enc_ER_dir))
    Gen.dec.load_state_dict(util.del_extra_keys(par_dec_dir))
    Dis_ER_val.enc.load_state_dict(util.del_extra_keys(par_Enc_ER_dir))
    Dis_ER_val.fc.load_state_dict(util.del_extra_keys(par_fc_ER_dir))

    Dis_ER.load_state_dict(util.del_extra_keys(par_Dis_ER_dir))

    CE = nn.CrossEntropyLoss()

    print('---- testing ----')
    with torch.no_grad():
        Dis_ER_val.eval()
        accuraccy, ce = Val_acc(expr_tt_loader, Dis_ER_val, CE, device)
        print(accuraccy)


    with torch.no_grad():
        Dis_ER.eval()
        accuraccy, ce= Val_acc(expr_tt_loader, Dis_ER, CE, device)
        print(accuraccy)


        
if __name__ == '__main__':

    test()

    
    
