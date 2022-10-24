# 我们来实现拟合y = x*x，前0.75比例数据用于训练，剩下用于测试，适当加入噪声
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#x = np.linspace(0,math.pi*8,2000)

x = np.linspace(-5, 5, 100)

noise = np.random.uniform(-5,5,len(x))

#y = np.sin(x)
y = pow(x,2)+0.2*noise
plt.plot(x,y)



#利用前3个波浪线来预测第4个波浪线
#特征就是前5日的数值
# train = []
# label = []
# for i in range(len(y)-5):
#     train.append(y[i:i+5])
#     label.append(y[i+5])
# train = np.array(train)
# label = np.array(label)

train = x 
label = y

p = 0.75
X_train = train[:int(len(train)*p)]
y_train = label[:int(len(train)*p)]
X_test = train[int(len(train)*p):]
y_test = label[int(len(train)*p):]

# plt.plot(range(len(y_train)), y_train)

class Net_R(nn.Module):
    def __init__(self):
        super(Net_R, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.pre = nn.Linear(10, 1)
 
    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.pre(x)
        return x

NN_net = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# net = Net_R()

net = NN_net

criterion=nn.MSELoss() # 使用CrossEntropyLoss损失
optm=torch.optim.Adam(net.parameters(), lr=0.1) # Adam优化
epochs=200 # 训练1000次

feature_dims = 1

x=torch.from_numpy(X_train).float().reshape(-1,feature_dims)
# x =x.reshape(-1,1) #在只有一个特征维度的时候，必须加上这句话，否则会报错
# print(x.shape)
y=torch.from_numpy(y_train).float().reshape(-1,1)


for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    
    y_hat=net(x)
    loss=criterion(y_hat,y) # 计算损失
    optm.zero_grad() # 前一步的损失清零
    loss.backward() # 反向传播
    optm.step() # 优化
    if (i+1)%1 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(X_test).float()
        test_in = test_in.reshape(-1,feature_dims)
        test_l=torch.from_numpy(y_test).float()
        test_l = test_l.reshape(-1,1)
        test_out=net(test_in)
        #print("test_out:",test_out,test_out.shape)
        #print(test_out.max(-1)[0])
        # 使用我们的测试函数计算准确率
        accu=criterion(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Test loss：{:.2f}".format(i+1,loss.item(),accu))



plt.plot(range(len(test_l)),test_l,"r")
plt.scatter(range(len(test_out.detach())),test_out.detach(),c="g")







