import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

##1、数据导入
batch_size=64
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081))])

train_dataset=datasets.MNIST(root='',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
train_loader=DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
test_dataset=datasets.MNIST(root='',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
test_loader=DataLoader(dataset=test_dataset,
                       shuffle=False,
                       batch_size=batch_size)

class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.conv1x1=nn.Conv2d(in_channels,16,kernel_size=1)

        self.conv5x5_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.conv5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2,stride=1)

        self.conv3x3_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.conv3x3_2=nn.Conv2d(16,24,kernel_size=3,padding=1,stride=1)
        self.conv3x3_3=nn.Conv2d(24,24,kernel_size=3,padding=1,stride=1)

        self.pool_branch=nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        conv1x1=self.conv1x1(x)

        conv5x5=self.conv5x5_2(self.conv5x5_1(x))

        conv3x3=self.conv3x3_3(self.conv3x3_2(self.conv3x3_1(x)))

        pool_branch=self.pool_branch(F.avg_pool2d(x,kernel_size=3,padding=1,stride=1))

        outputs=[conv1x1,conv3x3,conv5x5,pool_branch]

        return torch.cat(outputs,dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.inception1=Inception(in_channels=10)
        self.inception2=Inception(in_channels=20)

        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(1408,10)

    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        x=self.inception1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.inception2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)

        return x

model=Net()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,target=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d,%5d] loss:%.3f' %(epoch+1,batch_idx+1,running_loss) )
            running_loss=0.0
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('accuracy on test set:%.2f %%' %(100*correct/total))
    return correct/total

if __name__=='__main__':
    epoch_list=[]
    acc_list=[]
    for epoch in range(10):
        train(epoch)
        acc=test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list,acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


