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


