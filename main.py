import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size = 1):
        super(MyLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        # 此处input_size是我们word2vec的词向量的维度；
        # 这里设置了输入的第一个维度为batchsize，那么在后面构造输入的时候，需要保证第一个维度是batch size数量
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size) # 输出层，一个值
        self.activ = nn.Sigmoid() # 激活函数
    def init_hidden(self, batch_size):  # 初始化两个隐藏向量h0和c0
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim).to(device)),
                Variable(torch.zeros(1, batch_size, self.hidden_dim).to(device)))

    def forward(self, input):
        # 重新初始化隐藏状态
        self.hidden = self.init_hidden(input.size(0))  # input.size(0)得到batch_size
        # lengths 是每个序列的实际长度，用于padding
        lengths = get_lengths(input)
        # Pack the padded sequence
        packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        # Unpack the packed sequence
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Get the output from the last valid time step for each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2)).unsqueeze(1)
        last_out = lstm_out.gather(1, idx).squeeze(1)
        out = self.fc(last_out)
        out = self.activ(out)
        return out

class TextDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        # 应用变换
        if self.transform:
            text = self.transform(text)

        return text, label

def get_lengths(inputs):
    # inputs shape: (batch_size, sequence_length, vector_size)
    # 如果 padding 是全 0，则计算非零行的数量即为每个序列的长度
    non_zero_mask = torch.sum(inputs, dim=2) != 0
    # Count non-zero rows for each batch item
    lengths = torch.sum(non_zero_mask, dim=1)
    lengths = lengths.to(device)
    lengths = lengths.long()
    return lengths

# 文本到向量的转换(Word2Vec)
class TextToTensor:
    def __call__(self, text):
        # 将文本转换为词向量的数组
        vectors = []
        words = preprocess_text(text)
        for word in words:
            if word not in tranModel.wv:
                continue
            vector = tranModel.wv[word]
            assert len(vector) == vector_size
            vectors.append(torch.tensor(vector))
        return vectors

#
def read_data(root_dir):
    data = []
    labels = []
    # 遍历负面评论文件夹
    neg_dir = os.path.join(root_dir, 'neg')
    for filename in os.listdir(neg_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()  # 从文件中读取内容并去除首尾空白字
                data.append(content)
                labels.append(0)  # 负面评论标记为0
    # 遍历正面评论文件夹
    pos_dir = os.path.join(root_dir, 'pos')
    for filename in os.listdir(pos_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                data.append(content)
                labels.append(1)  # 正面评论标记为1
    return data, labels


# 文本预处理
def preprocess_text(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 分词
    words = word_tokenize(text.lower())
    # 去除停用词
    words = [word for word in words if word not in stop_words]
    return words

# 向量调整 shape
def collate_fn(sample_batch):
    # sample_batch 是 一个 batch 的 (texts, labels) 样本
    texts, labels = zip(*sample_batch)
    # 填充，将 text 从 list 转为 pytorch_tensor
    lengths = [len(text) for text in texts]
    max_length = max(lengths)

    padding_value = torch.zeros(vector_size)
    padded_texts = []
    for text in texts:
        padded_text = text + [padding_value] * (max_length - len(text))
        padded_texts.append(torch.stack(padded_text))

    padded_texts = torch.stack(padded_texts)
    # 将标签转换为形状 (batch_size, 1)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    return padded_texts, labels

# 训练函数
def train(dataloader, model, loss_fn, optimizer):  # 模型训练过程的定义；这个可以看作是模板，以后写pytorch程序基本都这样
    size = len(dataloader.dataset)
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):  # 这里是dataloader的迭代器，每次迭代都会返回一个batch的数据
        # Compute prediction and loss
        X, y = X.to(device), y.to(device) # 把数据放到GPU上

        # Compute prediction error
        pred = model(X) # 这里是模型预测
        loss = loss_fn(pred, y)  # 计算loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 验证函数
def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 设置模型为评估模式
    val_loss, correct = 0, 0
    with torch.no_grad():  # 不计算梯度
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # 将数据转移到GPU（如果可用）
            pred = model(X)
            val_loss += loss_fn(pred, y).item()  # 计算损失
            predicted_labels = (pred > 0.5).type(torch.float)  # 二分类情况下的阈值
            correct += (predicted_labels == y).type(torch.float).sum().item()  # 计算准确率
    val_loss /= num_batches  # 计算平均损失
    val_accuracy = correct / size  # 计算准确率
    print(f"Validation Error: \n Accuracy: {(100 * val_accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")

# 测试函数
def test(dataloader, model, loss_fn):  # 模型测试过程的定义，这个也是模板，以后可以借鉴
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 设置模型为测试模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(torch.sigmoid(pred))
            test_loss += loss_fn(pred, y).item()
            predicted_labels = (torch.sigmoid(pred) > 0.5).type(torch.float)
            correct += (predicted_labels == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # 超参数
    vector_size = 128# 定义词向量长度
    hidden_dim = 64 # 隐藏层维度
    epochs = 10 # 训练的轮数
    batch_size = 256 # 每次训练的样本数量

    # 设备选择
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 定义路径
    train_dir_path = f'G:/0Codes/nju_codes/24spr/LLM/DL24/DataSet/bbc/train'
    eval_dir_path = f'G:/0Codes/nju_codes/24spr/LLM/DL24/DataSet/bbc/eval'
    test_dir_path = f'G:/0Codes/nju_codes/24spr/LLM/DL24/DataSet/bbc/test'
    # 获取英语停用词列表
    stop_words = set(stopwords.words('english'))

    # 读取数据
    (train_data, train_labels) = read_data(train_dir_path)
    (eval_data, eval_labels) = read_data(eval_dir_path)
    (test_data, test_labels) = read_data(test_dir_path)
    print("Read Data Ready!")
    # 生成词典 common_texts
    train_dataset = TextDataset(train_data, train_labels)
    train_dataset = TextDataset(train_data, train_labels)
    test_dataset = TextDataset(test_data, test_labels)
    comments = train_dataset.data # comments = [train_dataset.data, test_dataset.data]
    common_texts = [preprocess_text(comment) for comment in comments]

    # 构建 Word2Vec模型，保存模型
    tranModel = Word2Vec(sentences=common_texts, vector_size=vector_size, window=5, min_count=1, workers=4)
    tranModel.save('word2vec.model')

    # 将每一个评论转换为词向量的数组，创建数据集
    text_to_tensor = TextToTensor()
    train_data = TextDataset(train_data, train_labels, transform=text_to_tensor)
    eval_data = TextDataset(eval_data, eval_labels, transform=text_to_tensor)
    test_data = TextDataset(test_data, test_labels, transform=text_to_tensor)
    print("Data Set Ready!")

    # Create data loaders.这个也是标准用法，只要按照要求自定义数据集，就可以用标准的dataloader加载数据
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn).to(device)
    eval_dataloader = DataLoader(eval_data, batch_size=32, shuffle=True, collate_fn=collate_fn).to(device)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn).to(device)
    print("Data Loader Ready!")

    # 创建一个神经网络模型对象并将它移动到指定的设备
    model = MyLSTM(vector_size, hidden_dim).to(device)

    loss_fn = nn.BCELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    # 下面这个训练和测试的过程也是标准形式，我们用自己的数据也还是这样去写
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        eval(eval_dataloader, model, loss_fn)
    print("Done!")

    # 模型可以保存下来，这里model文件夹要和当前py文件在同一个目录下
    torch.save(model.state_dict(), "model/lstm")
    print("Saved PyTorch Model State to the project root folder!")

    print("Testing-------------------------------")
    test(test_dataloader, model, loss_fn)