# AutoTrainCNN

## Prerequisites

1. python+pytorch+cifar10

## Setting

1. 把cifar10数据下载到data文件夹下
2. 把想要训练的网络连接方式文件放到net文件下
3. 配置网络连接方式（见文末）

## Document

这个代码可以通过配置网络连接方式，然后生成相应的网络模型进行训练。大大的缩短了训练者的学习成本。

**网络连接方式**

网络文件名字:
    0.0file.txt 0.1file.txt 0.2file.txt 0.3file.txt 0.4file.txt
网络文件内容：

    0 1 1 0 0 1 0 
    0 0 0 1 0 1 0 
    0 0 0 0 1 0 0 
    0 0 0 0 0 0 1 
    0 0 0 0 0 0 1 
    0 0 0 0 0 0 1 
    0 0 0 0 0 0 0 

第n行m列上的数字代表第n个节点和第m个节点的连接方式。1是连接，0是不连接。
本代码只支持5个stage的网络连接。因此网络文件为5个。