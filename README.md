# Machine-Learning
# conda到底是什么 
Conda是开源软件包管理系统和环境管理系统。Conda可以快速安装、运行和更新软件包。在功能上Conda可以看作是pip 和 vitualenv 的组合。
命令提示符中：

激活conda
`activate`

推出conda
`deactivate`

查看版本
`conda -V`

创建一个新的虚拟环境
`conda create -n py37 python=3.7`

查看cuda位置
`set cuda`

查看cuda版本
`nvcc -V`

查看有多少虚拟环境
`conda env list`

进入虚拟环境后，查看虚拟环境里有多少包
`pip list`

## 使用jupyter建立一个新的虚拟环境
### 1、配置新的虚拟环境
创建
`conda create -n your_env_name python=3.7`

打开
`conda activate your_env_name`

### 2、建立连接

在虚拟环境中下载这个包
`pip install ipykernel`

链接
`python -m ipykernel install --user your_env_name --name `

之后重启jupyter

### 3、查看当前环境
`import sys`
`print(sys.version)`
`print(sys.executable)`

# 神经网络
## 前向神经网络

W的列数（即`W.shape[1]`）就是下一层的units数（即下一层的神经元数量）
W的行数就是前一层的神经元数量，即输入时候的数量



### tensorflow一般步骤
1、指定模型，告诉tf如何工作

`mdoel=Sequential([Dense(units=25,activation='sigmoid'),
Dense(units=15,activation='sigmoid'),
Dense(units=1,activation='sigmoid')
]`

2、编译模型  

选择合适的损失函数
`model.compile(loss=BinaryCrossentropy())`

3、训练模型  

epochs指的是梯度下降的次数
`model.fit(train_x,trian_y,epochs=100)`

## 卷积层
### 卷积层是啥
不是所有的输入都能被下一层的神经元接收到，可能只能看到一部份
### 好处
1、运算速度更快

2、训练时，需要的训练数据更少，不容易过拟合


