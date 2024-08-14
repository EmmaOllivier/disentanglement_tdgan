'''
some functions used in the model
'''
import numpy as np
from torch.autograd import Variable
import PIL.Image as Image
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import pandas as pd


def Val_acc_train(loader, Dis, criterion, device, e, epoch):

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

        val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
        loop_test.set_description(f"Epoch [{e}/{epoch}] training")
        loop_test.set_postfix(accuracy_pain=val_acc*100)

    val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)

    val_acc_pain = (np.sum(((GT_list == pre_list) & (GT_list == 1 )).astype(float)) / (np.sum((GT_list == 1).astype(float))))
    print("val_acc_pain : "+ str(val_acc_pain))

    val_acc_no_pain = (np.sum(((GT_list == pre_list) & (GT_list == 0 )).astype(float)) / (np.sum((GT_list == 0).astype(float))))
    print("val_acc_no_pain : "+str(val_acc_no_pain))

    return val_acc, val_acc_pain, val_acc_no_pain, val_ce

def Val_acc(loader, Dis, criterion, device, e, epoch):
    pre_list = []
    GT_list = []
    val_ce = 0
    loop_test = tqdm(loader ,colour='GREEN')
    for i, (batch_FR_x_r, batch_FR_y_r, batch_FR_y_pain_r, batch_ER_x_r, batch_ER_y_r) in enumerate(loop_test):
        GT_list = np.hstack((GT_list, batch_ER_y_r.numpy()))
        batch_ER_x_r = Variable(batch_ER_x_r).to(device)
        batch_ER_y_r = Variable(batch_ER_y_r).to(device)
        _, batch_p = Dis(batch_ER_x_r)

        batch_result = batch_p.cpu().data.numpy().argmax(axis=1)
        pre_list = np.hstack((pre_list, batch_result))
        val_ce += criterion(batch_p, batch_ER_y_r).cpu().data.numpy()
        val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
        loop_test.set_description(f"Epoch [{e}/{epoch}] training")
        loop_test.set_postfix(accuracy_pain=val_acc*100)

    val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
    val_ce = val_ce / i

    return val_acc, val_ce


def combinefig_dualcon(FR_mat, ER_mat, Fake_mat, con_FR, con_ER, save_num=3):

    save_num = min(FR_mat.shape[0], save_num)
    imgsize = np.shape(FR_mat)[-1]
    img = np.zeros([imgsize * save_num, imgsize * 5, 3])
    for i in range(0, save_num):
        img[i * imgsize: (i + 1) * imgsize, 0 * imgsize: 1 * imgsize, :] = FR_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 1 * imgsize: 2 * imgsize, :] = ER_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 2 * imgsize: 3 * imgsize, :] = Fake_mat[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 3 * imgsize: 4 * imgsize, :] = con_FR[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 4 * imgsize: 5 * imgsize, :] = con_ER[i, :, :, :].transpose([1, 2, 0])

    return img


def preprocess_img(img_dir, device):
    img = Image.open(img_dir).convert('L').resize((128, 128))
    img = torch.from_numpy(np.array(img)/255).unsqueeze(0).unsqueeze(0).float()
    img = Variable(img).to(device)
    return img



def Val_acc_single(x_ER, Dis, device, name):
    exprdict = {
        0: 'Level0',
        1: 'Level4',
    }
    x_ER = Variable(x_ER).to(device)
    _, x_p = Dis(x_ER)
    pred_cls = x_p.cpu().data.numpy().argmax(axis=1).item()
    print('the predicted class of model {} is: {}'.format(name, exprdict[pred_cls]))


def del_extra_keys(model_par_dir):
    # the pretrained model is trained on old version pytorch, some extra keys should be deleted before loading
    model_par_dict = torch.load(model_par_dir)
    model_par_dict_clone = model_par_dict.copy()
    # delete keys
    for key, value in model_par_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_par_dict[key]
    
    return model_par_dict


class data_augm:
    def __init__(self,resolution):
        self.H_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.Jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = self.H_flip(x)
        x = self.Jitter(x)
        x = x/255
        return x

class data_adapt:
    def __init__(self,resolution):
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = x/255
        return x
    
