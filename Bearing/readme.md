## 背景
在真实的工业生产环境下，工业设备往往处于不同的工作状态，比如轴承的转速发生改变，刀具切削的物件改变，这些工况的改变会导致数据分布的变化。因此用于训练预测模型的数据和实际测试的数据分布可能不一致，甚至训练集和测试集分别包含多种工况，这影响了剩余使用寿命预测模型的泛化性能。

提出了一种新的解决多工况条件下领域自适应的模型DASL(Doamin Adaptation based on SAE-LSTM)。



## 模型结构
模型包括两部分：稀疏自编码器(SAE)和长短期记忆网络(LSTM)。利用稀疏自编码器(SAE)作为特征提取层，将原始特征(源域和目标域)映射到了新的特征空间，利用最大均值差异(MMD)减小源域和目标域的分布差异，并利用多层网络的非线性信息深入挖掘源数据中的特征；工业数据中多为时间序列数据，考虑到样本之间的时间依赖特性，我们采用了LSTM进行时间序列预测，并在最后一层加入全连接层作为输出，实现了端到端 (End-to-End) 的工业故障诊断与预测模型。注意模型同时输入源域和目标域。

## 软件环境
框架：Pytorch

语言：Python（3.7.3）

配置：GPU（12G）

相关包版本：
conda 4.7.12

conda-build 3.18.8

conda-package-handling 1.3.11

conda-verify 3.4.2

flake8 3.7.9

h5py 2.9.0

imageio 2.5.0

jupyter 1.0.0

jupyterlab 1.0.2

numpy 1.16.4

numpydoc 0.9.1

pandas 0.24.2

pep8 1.7.1

pip 19.1.1

pylint 2.3.1

scikit-image 0.15.0

scikit-learn 0.21.2

scipy 1.3.0

seaborn 0.9.0

tensorboard 1.14.0

## 代码

**目录树**

![image](https://github.com/minelabwot/phmlearn/blob/master/Bearing/image/Tree.png)


**调用**

首先对原始数据进行特征提取

```
python feature_extraction.py data='./home/train.cav' save=保存路径
```

然后对稀疏自编码器进行预训练

```
python SAE.py train='./home/train.cav' save=保存路径
```

预训练结束后对整个DASL模型进行训练

```
python DASL_model.py train='./home/train.cav' test='./home/test.cav'
```

具体代码详解可参考：http://www.phmlearn.com/u/13212127650/blogs/15#wow14


## 实验结果





