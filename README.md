## 实验报告

### 数据集划分与统计

- 训练集：25000 条，验证集：3600条，验证集：7200条，比例大约为 7:1:2
- 训练集、验证集、验证集 特征
  - 各自有 消极（neg)、积极（pos）评论，各占子集的 1/2
  - 三个子集互相独立，没有重复评论
  - 三个子集的评论所包含的电影组各自不相交

划分方式描述，以及上述数据集的特征统计

### 实验结果

- 实验代码：[1193600423/PytorchProject (github.com)](https://github.com/1193600423/PytorchProject)
- 超参数设置

```python
vector_size = 128# 定义词向量长度
hidden_dim = 128 # 隐藏层维度
epochs = 10 # 训练的轮数
batch_size = 256 # 每次训练的样本数量
loss_fn = nn.BCEWithLogitsLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器
self.activ = nn.Sigmoid() # 激活函数
```

- 训练结果

```
Using cuda device
Read Data Ready!
Data Set Ready!
Data Loader Ready!
Epoch 1
-------------------------------
loss: 0.748840  [  256/25000]
Validation Error: 
 Accuracy: 58.0%, Avg loss: 0.665627 

Epoch 2
-------------------------------
loss: 0.667073  [  256/25000]
Validation Error: 
 Accuracy: 66.9%, Avg loss: 0.635128 

Epoch 3
-------------------------------
loss: 0.635156  [  256/25000]
Validation Error: 
 Accuracy: 78.9%, Avg loss: 0.604506 

Epoch 4
-------------------------------
loss: 0.609163  [  256/25000]
Validation Error: 
 Accuracy: 73.7%, Avg loss: 0.611582 

Epoch 5
-------------------------------
loss: 0.617042  [  256/25000]
Validation Error: 
 Accuracy: 79.6%, Avg loss: 0.595800 

Epoch 6
-------------------------------
loss: 0.609618  [  256/25000]
Validation Error: 
 Accuracy: 81.2%, Avg loss: 0.587422 

Epoch 7
-------------------------------
loss: 0.574522  [  256/25000]
Validation Error: 
 Accuracy: 80.6%, Avg loss: 0.584543 

Epoch 8
-------------------------------
loss: 0.604193  [  256/25000]
Validation Error: 
 Accuracy: 82.4%, Avg loss: 0.583190 

Epoch 9
-------------------------------
loss: 0.584976  [  256/25000]
Validation Error: 
 Accuracy: 82.3%, Avg loss: 0.578638 

Epoch 10
-------------------------------
loss: 0.593195  [  256/25000]
Validation Error: 
 Accuracy: 84.7%, Avg loss: 0.579751 

Done!
Saved PyTorch Model State to the project root folder!
Testing-------------------------------
Test Error: 
 Accuracy: 84.8%, Avg loss: 0.580702 
```