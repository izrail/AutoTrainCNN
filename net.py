# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:59:43 2020

@author: izrail
"""
# coding=utf-8
import sys
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from thop import profile
graph=[]
#print(sys.argv[1]+"\n")
for i in range(5):
    a = numpy.loadtxt("./python/net/4"+"."+str(i)+"file.txt")
    graph.append(a.astype(np.int32))

# torchvision输出的是PILImage，值的范围是[0, 1]。
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                                ])
 
# 训练集，将绝对目录/data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中
trainset = torchvision.datasets.CIFAR10(root='E:\\python\\data', train=True,transform=transform)
 
# 将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
#num_workers改为0解决了无法异步提交多个个体问题
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
initNet = [64, 64, 64, 64, 64, 64, 64, 'M', 128 ,128 ,128 ,128 ,128 ,128 , 128, 'M',
         256, 256 ,256 ,256 ,256 ,256 ,256 , 256, 'M', 512, 512 ,512 ,512 ,512 ,512 ,512 , 512, 'M',
          512, 512 ,512 ,512 ,512 ,512 ,512 , 512, 'M']
initNum = []
class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__()  # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.convList=[]
        preLayer = 3
        layerNum = 0
        for i,layer in enumerate(initNet):
            if layer =='M':
                pass
                initNum.append(layerNum)
                layerNum = 0
            else:
                self.convList.append(nn.Conv2d(preLayer,layer,3,padding=1))
                layerNum = layerNum + 1
                preLayer = layer

        
        self.norm64 = nn.BatchNorm2d(64)
        self.norm128 = nn.BatchNorm2d(128)
        self.norm256 = nn.BatchNorm2d(256)
        self.norm512 = nn.BatchNorm2d(512)
        self.avgPool2d=nn.AvgPool2d(kernel_size=1,stride=1)
        self.fc=nn.Linear(512,10)

        
    #修改网络模型
    def modify(self):
        stageNum = 0
        index = 0
        distance = 0
        for i, layer in enumerate(initNet):
            '''for i in range(7):
                myCount = 0
                for j in range(7):
                    myCount +=graph[stageNum][j][i]
                count.append(myCount)'''
            if index == 0:
                index = index + 1 
                continue
            count = 0 
            for j in range(7):
                count += graph[stageNum][j][index]

            if count == 0:
                self.convList[i - stageNum] = nn.Conv2d(layer,layer,3,padding = 1)
            else:
                self.convList[i - stageNum] = nn.Conv2d(layer*(count - 1) + layer,layer,3,padding = 1)
            if layer == 'M':
                stageNum = stageNum + 1
                index = 0
            else:
                index = index + 1
        '''count=[]
        for i in range(7):
            myCount=0
            for j in range(7):
                myCount+=graph[0][j][i]
            count.append(myCount)
        
        for i in range(1,7):
            if count[i] == 0 :
                self.convList[i] = nn.Conv2d(64,64,3,padding=1)
            else:
                self.convList[i] = nn.Conv2d(64*(count[i]-1)+64,64,3,padding=1)
                #print("stage1:",64*(count[i]-1)+64)
     
        count=[]
        for i in range(7):
            myCount=0
            for j in range(7):
                myCount+=graph[1][j][i]
            count.append(myCount)
            
        for i in range(1,7):
            if count[i] == 0 :
                self.convList[i+7] = nn.Conv2d(128,128,3,padding=1)
            else:
                self.convList[i+7] = nn.Conv2d(128*(count[i]-1)+128,128,3,padding=1)
                #print("stage2:",64*(count[i]-1)+64)
    
        count=[]
        for i in range(7):
            myCount=0
            for j in range(7):
                myCount+=graph[2][j][i]
            count.append(myCount)
        for i in range(1,7):
            if count[i] == 0 :
                self.convList[i+14] = nn.Conv2d(256,256,3,padding=1)
            else:
                self.convList[i+14] = nn.Conv2d(256*(count[i]-1)+256,256,3,padding=1)
  
        count=[]
        for i in range(7):
            myCount=0
            for j in range(7):
                myCount+=graph[3][j][i]
            count.append(myCount)
        for i in range(1,7):
            if count[i] == 0 :
                self.convList[i+22] = nn.Conv2d(512,512,3,padding=1)
            else:
                self.convList[i+22] = nn.Conv2d(512*(count[i]-1)+512,512,3,padding=1)
  
        count=[]
        for i in range(7):
            myCount=0
            for j in range(7):
                myCount+=graph[4][j][i]
            count.append(myCount)
        for i in range(1,7):
            if count[i] == 0 :
                self.convList[i+30] = nn.Conv2d(512,512,3,padding=1)
            else:
                self.convList[i+30] = nn.Conv2d(512*(count[i]-1)+512,512,3,padding=1)
                #print("stage1:",512*(count[i]-1)+512)'''


    def forward(self, x):
        xList=[]
        xList.append(F.relu(self.norm64(self.convList[0](x))))
        
        for i in range(1,7):
            flag=0
            #print("i=:",i)
            for j in range (0,i):
                #print(j,",",flag,"\n")
                if graph[0][j][i] == 1:
                    if flag == 0:
                        x=xList[j]
                    else:
                        x=torch.cat((x,xList[j]),1)
                    flag=flag+1
            if flag == 0:
#                xList.append(F.relu(self.norm64(self.convList[i](x))))
                xList.append("test")
                continue
            #print("x size: ",x.size())
            if i == 6 :
                xList.append(F.max_pool2d(F.relu(self.norm64(self.convList[i](x))),(2,2)))
            else:
                xList.append(F.relu(self.norm64(self.convList[i](x))))
        x=xList[-1]
        
        xList=[]
        xList.append(F.relu(self.norm128(self.convList[7](x))))
        
        for i in range(1,7):
            flag=0
            for j in range (0,i):
                if graph[1][j][i] == 1:
                    if flag == 0:
                        x=xList[j]
                    else:
                        x=torch.cat((x,xList[j]),1)
                    flag=flag+1
            if flag == 0:
               # xList.append(F.relu(self.norm128(self.convList[i+7](x))))
                xList.append("test")
                continue
            #print("stage2,x size: ",x.size())
            if i == 6 :
                xList.append(F.max_pool2d(F.relu(self.norm128(self.convList[i+7](x))),(2,2)))
            else:
                xList.append(F.relu(self.norm128(self.convList[i+7](x))))
        x=xList[-1]
        
        
        xList=[]
        xList.append(F.relu(self.norm256(self.convList[14](x))))
        
        for i in range(1,7):
            flag=0
            for j in range (0,i):
                if graph[2][j][i] == 1:
                    if flag == 0:
                        x=xList[j]
                    else:
                        x=torch.cat((x,xList[j]),1)
                    flag=flag+1
            if flag == 0:
                #xList.append(F.relu(self.norm256(self.convList[i+14](x))))
                xList.append("test")
                continue
            #print("x size: ",x.size())
            if i == 6 :
                xList.append(F.relu(self.norm256(self.convList[i+14](x))))
            else:
                xList.append(F.relu(self.norm256(self.convList[i+14](x))))
        xList.append(F.max_pool2d(F.relu(self.norm256(self.convList[21](xList[-1]))),(2,2)))
        x=xList[-1]
  
     
        xList=[]
        xList.append(F.relu(self.norm512(self.convList[22](x))))
        
        for i in range(1,7):
            flag=0
            for j in range (0,i):
                if graph[3][j][i] == 1:
                    if flag == 0:
                        x=xList[j]
                    else:
                        x=torch.cat((x,xList[j]),1)
                    flag=flag+1
            if flag == 0:
                #xList.append(F.relu(self.norm512(self.convList[i+22](x))))
                xList.append("test")
                continue
            #print("x size: ",x.size())
            if i == 6 :
                xList.append(F.relu(self.norm512(self.convList[i+22](x))))
            else:
                xList.append(F.relu(self.norm512(self.convList[i+22](x))))
        xList.append(F.max_pool2d(F.relu(self.norm512(self.convList[29](xList[-1]))),(2,2)))
        x=xList[-1]
        
        xList=[]
        xList.append(F.relu(self.norm512(self.convList[30](x))))
        
        for i in range(1,7):
            flag=0
            for j in range (0,i):
                if graph[4][j][i] == 1:
                    if flag == 0:
                        x=xList[j]
                    else:
                        x=torch.cat((x,xList[j]),1)
                    flag=flag+1
            if flag == 0:
                #xList.append(F.relu(self.norm512(self.convList[i+30](x))))
                xList.append("test")
                continue
            #print("x size: ",x.size())
            if i == 6 :
                xList.append(F.relu(self.norm512(self.convList[i+30](x))))
            else:
                xList.append(F.relu(self.norm512(self.convList[i+30](x))))
        xList.append(F.max_pool2d(F.relu(self.norm512(self.convList[37](xList[-1]))),(2,2)))
        x=xList[-1]
       
        
        x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
       
        x = self.fc(x)  # 输入x经过全连接3，然后更新x
        return x
 
    # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
    def num_flat_features(self, x):
        size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。
        # 【1:】让我们把注意力放在后3维上面,是因为 x.size() 会 return [nSamples, nChannels, Height, Width]。我们只需要展开后三项成为一个一维的 tensor。
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
 
    
net = Net()

#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
testset = torchvision.datasets.CIFAR10(root='E:\\python\\data', train=False, transform=transform)
 
# 将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

criterion = nn.CrossEntropyLoss()#叉熵损失函数

#for myNum in range(populationNum):
net.modify()

input = torch.randn(4,3,32,32)
flops,params = profile(net,inputs=(input,))

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
    
for epoch in range(1):  # 遍历数据集
 
    running_loss = 0.0
        # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0):
            # get the inputs
            #data is list
            # print(data[0].size())
            # print(i)
        inputs, labels = data  # data的结构是：[4x3x32x32的张量,长度4的张量]
            # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)  # 把input数据从tensor转为variable
            # print(input.grad_fn)
            # zero the parameter gradients
        optimizer.zero_grad()  # 将参数的grad值初始化为0
 
            # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 将output和labels使用叉熵计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 用SGD更新参数
 
            # 每200批数据打印一次平均loss值
        running_loss += loss.item()  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0  或使用loss.item()
        if i % 200 == 199:  # 每200批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    print('Finished Training')
 
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
           
        value, predicted = torch.max(outputs.data, 1)  # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        
        total += labels.size(0)
        correct += (predicted == labels).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
print('Accuracy: %d %%' % (100 * correct / total))
file = open("/result/"+sys.argv[1]+"result.txt","w")
file.write("Accuracy:"+correct/total)
file.write("Flops:"+flops)
 
file.close()


