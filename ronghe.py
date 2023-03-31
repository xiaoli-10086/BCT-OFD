
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model_v3 import mobilenet_v3_small as model1
from mpvit import mpvit_tiny as model2

import math

#for x in model2.named_children():
    #print(x)

# 对应论文中的non-linear
class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x
class CoorAttention(nn.Module):
    def __init__(self,in_channels, out_channels, reduction = 32):
        super(CoorAttention, self).__init__()
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.poolw = nn.AdaptiveAvgPool2d((1,None))
        middle = max(8, in_channels//reduction)
        self.conv1 = nn.Conv2d(in_channels,middle,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(middle)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(middle,out_channels,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x): # [batch_size, c, h, w]
        identity = x
        batch_size, c, h, w = x.size()  # [batch_size, c, h, w]
        # X Avg Pool
        x_h = self.poolh(x)    # [batch_size, c, h, 1]

        #Y Avg Pool
        x_w = self.poolw(x)    # [batch_size, c, 1, w]
        x_w = x_w.permute(0,1,3,2) # [batch_size, c, w, 1]

        #following the paper, cat x_h and x_w in dim = 2，W+H
        # Concat + Conv2d + BatchNorm + Non-linear
        y = torch.cat((x_h, x_w), dim=2)   # [batch_size, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))  # [batch_size, c, h+w, 1]
        # split
        x_h, x_w = torch.split(y, [h,w], dim=2)  # [batch_size, c, h, 1]  and [batch_size, c, w, 1]
        x_w = x_w.permute(0,1,3,2) # 把dim=2和dim=3交换一下，也即是[batch_size,c,w,1] -> [batch_size, c, 1, w]
        # Conv2d + Sigmoid
        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))
        # re-weight
        return identity * attention_h * attention_w

class Net(nn.Module):
    def __init__(self, modelc, modelt):
        super(Net, self).__init__()


        self.features = list(modelc.children())[0]
        self.avgpool = list(modelc.children())[1]
        self.convs = list(modelc.children())[3]

        self.stem = list(modelt.children())[0]
        self.patch_embed_stages = list(modelt.children())[1]
        self.mhca_stages = list(modelt.children())[2]
        # self.class_token = nn.Parameter(torch.zeros(1, 1，768))
        # self._process_input = list(vit.children())[0]
        # self.encoder = list(vit.children())[1]

        self.coorattention = CoorAttention(792,792)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(792, 396)
        self.fc2 = nn.Linear(396, 2)
        # self.dropout1 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1024, 8)

    def forward(self, input):
        x_list = []
        m=input
        for i in range(13):
            m = self.features[i](m)
            x_list.append(m)
        #x = self.features(input)
        x=m
        #x = self.avgpool(x)
        #x = x.reshape(x.size(0), -1)


        # Reshape and permute the input tensor
        y = self.stem(input)  # Shape : [B, C, H/4, W/4]
        x_list = []
        for idx in range(4):
            att_inputs = self.patch_embed_stages[idx](y)
            y = self.mhca_stages[idx](att_inputs)
            x_list.append(y)
        # Classifier "token" as used by standard language architectures
        #y = self.avgpool(y)
        #y = y.view(y.size(0), -1)

        output = torch.cat((x, y), 1)
        output1 = self.coorattention(output)
        output = output+output1
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output

def loader_model(num_classes, name, path):
    checkpoint = torch.load(path)
    if num_classes==2:
        if name=='mobilenetv3':
            model = model1(num_classes)
        elif name=='mpvit':
            model = model2(num_classes=num_classes)
    # checkpoint_dict = checkpoint.module.state_dict()
    # model.load_state_dict(checkpoint['model_state'], strict=False)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    # model.load_state_dict(checkpoint_dict)
    # model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model).module()
    print(model.load_state_dict(checkpoint,strict=False))
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    # model = model.cuda()
    # model.eval().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #optimizer.load_state_dict(checkpoint)
    #epoch = checkpoint['epoch']
    return model, optimizer
    config, unprased = get_config()
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    test_dataset = get_test_loader(config.predictdata_dir,config.batch_size,config.input_size,config.num_workers)

def getModel():
    path1 = "./weights/"
    path2 = "./weights/"
    modelc, _ = loader_model(2, "mobilenetv3", path1)
    modelt, _ = loader_model(2, "mpvit", path2)

    '''total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))'''


    '''sum_ = 0
    for name, param in vit.named_parameters():
        mul = 1
        for size_ in param.shape:
            mul *= size_  # 统计每层参数个数
        sum_ += mul  # 累加每层参数个数
        print('%14s : %s' % (name, param.shape))  # 打印参数名和参数数量
    # print('%s' % param)						# 这样可以打印出参数，由于过多，我就不打印了
    print('参数个数：', sum_)'''


    for index, p in enumerate(modelc.parameters()):
        p.requires_grad = False

    for index, p in enumerate(modelt.parameters()):
        p.requires_grad = False

    model = Net(modelc, modelt)
    return model

getModel()




