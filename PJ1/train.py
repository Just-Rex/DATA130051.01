import os
import torch
from torch.utils.tensorboard import SummaryWriter
from components.data import load_data_mnist
from components.model import MLP
import components.functional as F
from components.optim import SGD
import sys
from tqdm import tqdm
import configparser
import numpy as np
import pickle


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['TRAIN'] # 选择对应参数
    model_path = params['model_path'] # 训练参数存放路径
    log_dir = params['log_dir']
    image_path = params['image_path'] # 图像存放路径
    batch_size = int(params['batch_size']) # 批大小
    num_hiddens = int(params['num_hiddens']) # 隐藏单元数量
    num_class = int(params['num_class']) # 预测类别
    activation = params['activation'] # 使用不同的激活函数
    num_epochs = int(params['num_epochs']) # 训练轮数
    learning_rate = float(params['learning_rate']) # 学习率
    weight_decay = float(params['weight_decay']) # 正则化强度
    
    # Move models to GPU if CUDA is available. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    tb = SummaryWriter(log_dir=log_dir)
    
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Build data
    train_iter, test_iter, num_inputs = load_data_mnist(image_path, batch_size)

    # Build the models
    mlp = MLP(num_inputs, num_hiddens, num_class, activation).to(device)
    
    # Loss and optimizer
    loss = F.CrossEntropyLoss(mlp)
    optimizer = SGD(mlp, lr=learning_rate, decay=weight_decay)

    # Train the models
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        total_acc = []
        total_loss = []
        train_bar = tqdm(train_iter, file=sys.stdout)
        for X, y in train_bar:
            y = np.array(y).flatten()
            logits = mlp(X)
            L = loss(logits, y)
            y_pred = logits.argmax(axis=1)

            total_loss.append(L.item())
            total_acc.append((y_pred == y).mean())
            L.backward()
            optimizer.step()
            train_loss = sum(total_loss) / len(total_loss)
            train_acc = sum(total_acc) / len(total_acc)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, num_epochs, train_loss, train_acc)

        total_acc = []
        total_loss = []
        test_bar = tqdm(test_iter, file=sys.stdout)
        for X, y in test_bar:
            y = np.array(y).flatten()
            logits = mlp(X)
            L = loss(logits, y)
            y_pred = logits.argmax(axis=1)
            total_loss.append(L.item())
            total_acc.append((y_pred == y).mean())

            test_loss = sum(total_loss) / len(total_loss)
            test_acc = sum(total_acc) / len(total_acc)
            test_bar.desc = "test epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch + 1, num_epochs, test_loss, test_acc)

        tb.add_scalars('MLP', {'train_loss':train_loss, 'train_acc':train_acc, 'test_loss':test_loss, 'test_acc':test_acc}, epoch + 1)
        if test_acc > best_acc:
            with open(os.path.join(model_path, 'MLP-{}-{}.pkl'.format(batch_size, learning_rate)), 'wb') as f:
                pickle.dump(mlp, f)
    tb.close()
main()
