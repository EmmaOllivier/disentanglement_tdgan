
import os,glob
import pandas as pd
import torch
import torchvision
import random
import numpy as np
from tqdm import tqdm
import PIL


class Dataset_Biovid_image_binary_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,nb_fold=1,preload=False,loaded_resolution=224, seed=42):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.seed = seed
        self.reset()
        q_fold = 40//nb_fold
        self.fold = [[j for j in range(i*q_fold,(i+1)*q_fold)] for i in range(nb_fold)]

        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.dataframe.loc[p,'path'])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self,fold=None,keep=False):
        dataframe = pd.read_csv(self.PATH_ANOT)
        self.dataframe = dataframe
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe = self.dataframe[~ self.dataframe['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe = self.dataframe[self.dataframe['id_video'].isin(self.fold[fold])]

        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image, random_state=self.seed).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            try:
                self.img_loaded[i] = self.resize(torchvision.io.read_image(self.dataframe.loc[index,'path']))
            except :
                print(self.dataframe.loc[index,'path'])
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        video_tensor = self.dataframe.loc[index,'id_video']
        
        return img_tensor, pain_tensor, ID_tensor

class Dataset_Biovid_image_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,nb_fold=1,preload=False,loaded_resolution=224, seed=42):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.seed=seed
        self.reset()
        q_fold = 40//nb_fold
        self.fold = [[j for j in range(i*q_fold,(i+1)*q_fold)] for i in range(nb_fold)]

        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.dataframe.loc[p,'path'])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self,fold=None,keep=False):
        dataframe = pd.read_csv(self.PATH_ANOT)
        self.dataframe = dataframe
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe = self.dataframe[~ self.dataframe['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe = self.dataframe[self.dataframe['id_video'].isin(self.fold[fold])]

        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image, random_state=self.seed).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            try:
                self.img_loaded[i] = self.resize(torchvision.io.read_image(self.dataframe.loc[index,'path']))
            except :
                print(self.dataframe.loc[index,'path'])
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        
        return img_tensor, pain_tensor,ID_tensor
    

class DualTrainDataset(torch.utils.data.Dataset):


    def __init__(self,PATH_IMG,PATH_ANOT1,PATH_ANOT2,transform = lambda x: x,IDs = None,set_type = None,nb_image = None,nb_fold=1,preload=False,loaded_resolution=224, seed=42):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT1 = PATH_ANOT1
        self.PATH_ANOT2 = PATH_ANOT2
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.seed=seed
        
        self.reset()
        q_fold = 40//nb_fold
        self.fold = [[j for j in range(i*q_fold,(i+1)*q_fold)] for i in range(nb_fold)]
        
        dataframe1 = pd.read_csv(self.PATH_ANOT1)
        dataframe2 = pd.read_csv(self.PATH_ANOT2)
        N1 = len(dataframe1)
        N2 = len(dataframe2)
        self.seeds=self.tab_of_seeds(N2-1)
        self.lessN=len(dataframe1)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded1 = torch.zeros((N1,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image1 = {}
        self.img_loaded2 = torch.zeros((N2,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image2 = {}

        self.transf = torchvision.transforms.Compose([
            torchvision.transforms.Resize([loaded_resolution, loaded_resolution]),
            torchvision.transforms.ToTensor(),
        ])
        
        
    def __len__(self):
        return len(self.index_link1)
    
    def reset(self,fold=None,keep=False):
        dataframe1 = pd.read_csv(self.PATH_ANOT1)
        self.dataframe1 = dataframe1
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe1 = self.dataframe1[~ self.dataframe1['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe1 = self.dataframe1[self.dataframe1['id_video'].isin(self.fold[fold])]

        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe1['ID']))}
        if not self.nb_image is None:
            self.index_link1 = list(self.dataframe1.sample(self.nb_image, random_state=self.seed).index)
        else:
            self.index_link1 = list(self.dataframe1.index)

        dataframe2 = pd.read_csv(self.PATH_ANOT2)
        self.dataframe2 = dataframe2
        if not fold is None and fold < len(self.fold) and fold >=0:
            if not keep:
                self.dataframe2 = self.dataframe2[~ self.dataframe2['id_video'].isin(self.fold[fold])]
            else:
                self.dataframe2 = self.dataframe2[self.dataframe2['id_video'].isin(self.fold[fold])]
        if not self.nb_image is None:
            self.index_link2 = list(self.dataframe2.sample(self.nb_image, random_state=self.seed).index)
        else:
            self.index_link2 = list(self.dataframe2.index)

    def tab_of_seeds(self, nb_seeds):
        seeds=[]
        for i in range(nb_seeds):
            seeds.append(i)
            if(i==0):
                print(i)
        return seeds



    def __getitem__(self,idx):
        img_idx = idx % self.lessN
        index = self.index_link1[img_idx]
        try :
            i = self.dic_image1[index]
        except:
            i = len(self.dic_image1)
            self.dic_image1[index] = i
            self.img_loaded1[i] = self.resize((torchvision.io.read_image(self.dataframe1.loc[index,'path'])))
        
        img_tensor1 = self.img_loaded1[i]
        img_tensor1 = self.transform(img_tensor1)
        pain_tensor1 = self.dataframe1.loc[index,'pain']

        ID_tensor = self.dic_ID[self.dataframe1.loc[index,'ID']]
        index2=index
        #index2 = self.index_link2[idx]
        
        random.seed(self.seeds[index-1])
        index2 = random.randint(0, len(self.dataframe2)-1)
        #print(index2)
        while(self.dataframe2.loc[index2,'pain']==self.dataframe1.loc[index,'pain']):
                index2=random.randint(0, len(self.dataframe2)-1)

        try :
            i = self.dic_image2[index2]
        except:

            i = len(self.dic_image2)
            self.dic_image2[index2] = i
            try:
                self.img_loaded2[i] = self.resize((torchvision.io.read_image(self.dataframe2.loc[index2,'path'])))
            except :
                print(self.dataframe2.loc[index2,'path'])
        
        img_tensor2 = self.img_loaded2[i]
        img_tensor2 = self.transform(img_tensor2)

        pain_tensor = self.dataframe2.loc[index2,'pain']
        ID_tensor2 = self.dataframe2.loc[index2,'ID']

        return img_tensor1,ID_tensor, pain_tensor1, img_tensor2,pain_tensor, ID_tensor2
    
    
    

    
    def img_prepro(self, img_path, transf=None, RGB=False):
        if RGB:
            img = PIL.Image.open(img_path).convert('RGB')
        else:
            img = PIL.Image.open(img_path).convert('L')
        if transf is not None:
            img = transf(img)


        return img

    
