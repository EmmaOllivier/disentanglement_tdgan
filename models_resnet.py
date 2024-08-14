import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
from thop import profile


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class fc_layer_resnet18(nn.Module):
    def __init__(self, par=None, p=0.5, cls_num=10575):
        super(fc_layer_resnet18, self).__init__()

        self.act = nn.ReLU()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, cls_num)

        self.dropout = nn.Dropout(p=p)

        if par:
            fc_dict = self.state_dict().copy()
            fc_list = list(self.state_dict().keys())

            fc_dict[fc_list[0]] = par['module.fc.weight']
            fc_dict[fc_list[1]] = par['module.fc.bias']

            self.load_state_dict(fc_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, fea):
        # fc1: bsx8x8x128 -> bsx8192 -> bsx1024
        self.fc1_out = self.act(self.fc1(self.dropout(fea.view(fea.size(0), -1))))
        # fc2: bsx1024 -> bsx1024
        self.fc2_out = self.act(self.fc2(self.dropout(self.fc1_out)))
        # fc3: bsx1024 -> bsxcls_num
        self.fc3_out = self.fc3(self.fc2_out)
        return self.fc3_out

class resblock(nn.Module):

    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out



class decoder_resnet18(nn.Module):
    def __init__(self, Nz=100, Nb=3, Nc=128, GRAY=False):

        super(decoder_resnet18, self).__init__()

        self.Nz = Nz

        self.emb1 = nn.Sequential(*[
            nn.Conv2d(512*2 + Nz, Nc, 3, 1, 1),
            nn.ReLU(),
        ])
        self.emb2 = self._make_layer(resblock, Nb, Nc)

        self.us1 = nn.Sequential(*[
            nn.ConvTranspose2d(Nc, 512, 10, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
        ])
        self.us2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ])
        self.us3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ])
        self.us4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ])
        self.us5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        ])
        if GRAY:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(32, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])

    def _make_layer(self, block, num_blocks, n_chan):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(n_chan))
        return nn.Sequential(*layers)

    def forward(self, enc_FR, enc_ER, noise=None, device=None):

        fea_ER = enc_ER
        fea_FR = enc_FR

        if noise is not None:
            noise = noise
        else:
            noise = Variable(torch.rand(fea_ER.shape[0], self.Nz, 1, 1))

        
        if device is not None:
            noise = noise.to(device)

        if self.Nz == 0:
            emb_in = torch.cat((fea_ER, fea_FR), dim=1)
        else:
            emb_in = torch.cat((fea_ER, fea_FR, noise), dim=1)
        # embedding: bsx(256+Nz)x8x8 -> bsxNcx8x8
        self.emb1_out = self.emb1(emb_in)
        # bsxNcx8x8 -> bsxNcx8x8
        self.emb2_out = self.emb2(self.emb1_out)

        # bsxNcx8x8 -> bsx512x16x16
        self.us1_out = self.us1(self.emb2_out)
        # bsx512x16x16 -> bsx256x32x32
        self.us2_out = self.us2(self.us1_out)
        # bsx256x32x32 -> bsx128x64x64
        self.us3_out = self.us3(self.us2_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us4_out = self.us4(self.us3_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us5_out = self.us5(self.us4_out)
        # bsx64x128x128 -> bsxout_chanx128x128
        self.img = self.us6(self.us5_out)
        #print(self.img.shape)

        return self.img



class Dis_resnet18(nn.Module):

    def __init__(self, fc=None, GRAY=False, cls_num=2):
        super(Dis_resnet18, self).__init__()


        self.enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor= torch.nn.Sequential(*list(self.enc.children())[:-1])

        self.fc = fc_layer_resnet18(cls_num=cls_num)

    def forward(self, x_in):
        self.fea = self.feature_extractor(x_in)
        self.result = self.fc(self.fea)
        return self.fea, self.result





class Gen_resnet18(nn.Module):
    '''
    the class of generator
    '''
    def __init__(self, clsn_ER=7, Nz=100, Nb=3, GRAY=False):
        super(Gen_resnet18, self).__init__()

        self.enc_FR = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.enc_ER = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor_ER= torch.nn.Sequential(*list(self.enc_ER.children())[:-1])
        self.feature_extractor_FR = torch.nn.Sequential(*list(self.enc_FR.children())[:-1])
 
        self.fc_ER = fc_layer_resnet18(cls_num=clsn_ER)

        self.dec = decoder_resnet18(Nz=Nz, GRAY=GRAY, Nb=Nb)
        self.dec.apply(weights_init)

    def infer_FR(self, x_FR):
        fea_FR = self.feature_extractor_FR(x_FR)
        return fea_FR

    def infer_ER(self, x_ER):
        fea_ER = self.feature_extractor_ER(x_ER)
        result_ER = self.fc_ER(fea_ER)
        return fea_ER, result_ER
    
    def infer_ER_without_result(self, x_ER):
        fea_ER = self.feature_extractor_ER(x_ER)
        return fea_ER

    def gen_img(self, x_FR, x_ER, noise=None, device=None):
        self.fea_FR = self.infer_FR(x_FR=x_FR)
        self.fea_ER, self.result_ER = self.infer_ER(x_ER=x_ER)
        self.img = self.dec(enc_FR=self.feature_extractor_FR(x_FR), enc_ER=self.feature_extractor_ER(x_ER), device=device)
        return self.img

    def gen_img_withfea(self, fea_FR, fea_ER):
        self.enc_FR.ds4_out = fea_FR
        self.enc_ER.ds4_out = fea_ER
        self.img = self.dec(enc_FR=self.FR, enc_ER=self.enc_ER)
        return self.img


