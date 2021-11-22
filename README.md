# 流程

### 数据处理模块
根据不同类型的原始语料编写不同的代码，需要处理的事情：
对数据进行预处理，分词等操作保存到特定文件

### 构造词典
使用上步骤中生成的文件进行构建词典操作，通常保存到pickle文件

### 构造dataset数据
基于词典构造训练数据集:
__getitem__()函数返回list类型
collate_fn()将所有数据改成统一长度

### 搭建模型
模型的构建

### 训练模型
train_model

### 模型评估(预测)
evaluation


# corpus
语料库
### corpus/merge_data
处理后存放的数据路径
### corpus/origin
原始数据路径

# model
模型存放路径

# pre_corpus
处理数据的代码
TODO: 原始数据使用RASA格式，需要修改代码

# recall
召回模块


# 使用方法
- 配置需要
- 在pre_corpus中执行pre_origin_corpus.py程序
