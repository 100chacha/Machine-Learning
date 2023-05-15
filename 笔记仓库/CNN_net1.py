import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#1、数据导入
train_data=datasets.MNIST(root="./MNIST",
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())
train_loader=DataLoader(dataset=train_data,
                        shuffle=True,
                        batch_size=32)
test_data=datasets.MNIST(root="./MNIST",
                         train=False,
                         download=True,
                         transform=transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,
                       shuffle=True,
                       batch_size=32)


#2、设计模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)

        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=3, stride=3)

        self.relu=torch.nn.ReLU()

        self.fc1=torch.nn.Linear(40,20)
        self.fc2=torch.nn.Linear(20,10)

    def forward(self,x):
        batch_size=x.size(0)
        x = self.pooling1(self.relu((self.conv1(x))))
        x = self.pooling1(self.relu((self.conv2(x))))
        x = self.pooling2(self.relu((self.conv3(x))))
        x = x.view(batch_size,-1)
        x=self.fc1(x)
        x=self.fc2(x)

        return x


model=Net()
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)


#3、设定loss与optimizer
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#4、训练函数
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx %300==299:
            print('[%d,%5d] loss: %.3f' %(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

#5、测试函数
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data , dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()

        print('accuracy on test set:%d %%' %(100*correct/total))
    return correct / total

if __name__=='__main__':
    epoch_list=[]
    acc_list=[]

    for epoch in range(10):
        train(epoch)
        acc=test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    import os
    os.environ['KMP_DOUPLICATE_LIB_OK']='TRUE'
    plt.plot(epoch_list,acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


