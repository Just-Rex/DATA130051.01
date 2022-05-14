# import timm
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.AlexNet import AlexNet
from utils.data import cutout_data


def main():
    # 使用GPU训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建tensorboard实例
    tb = SummaryWriter(log_dir="./tensorboard/AlexNet-cutout1.5")

    # 图像预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 设置批处理大小和线程使用数量
    batch_size = 32
    nw = os.cpu_count()
    print('Using {} dataloader workers every process'.format(nw))


    train_dataset = datasets.CIFAR100(root='./data', train = True, download = True,transform = data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = nw)
    val_dataset = datasets.CIFAR100(root = './data', train = False, download = True, transform = data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = nw)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))


    # 创建神经网络
    net = AlexNet().to(device)
    # 设置超参
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0002)


    epochs = 100
    best_acc = 0.0
    save_path = './AlexNet-cutout.pth'
    for epoch in range(epochs):
        # 训练过程
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # 清空梯度
            cutout_images, cutout_labels = cutout_data(images, labels, 0.5)
            optimizer.zero_grad()
            outputs = net(cutout_images.to(device))
            loss = loss_function(outputs.to(device), cutout_labels.to(device))
            loss.backward()
            # 更新学习参数
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        train_loss = running_loss / (step+1)
        # 验证过程
        net.eval()
        train_acc_num = 0.0
        val_acc_num = 0.0
        # 验证过程不计算梯度
        with torch.no_grad():
            train_bar = tqdm(train_loader, file=sys.stdout)
            for data in train_bar:
                images, labels = data
                outputs = net(images.to(device))
                predicts = torch.max(outputs, dim=1)[1]
                d = 0
                if epochs % 20 == 0 and d == 0:
                    d += 1
                train_acc_num += torch.eq(predicts, labels.to(device)).sum().item()
                train_bar.desc = "train epoch[{}/{}]".format(epoch + 1, epochs)
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                images, labels = data
                outputs = net(images.to(device))
                predicts = torch.max(outputs, dim=1)[1]
                val_acc_num += torch.eq(predicts, labels.to(device)).sum().item()
                val_bar.desc = "val epoch[{}/{}]".format(epoch + 1, epochs)

        train_acc = train_acc_num / train_num
        val_acc = val_acc_num / val_num
        # tb.add_scalars("AlexNet", {'train_loss_baseline': train_loss, 'train_accuracy_baseline':  train_acc, 'val_accuracy_baseline':  val_acc}, epoch)
        tb.add_scalars("AlexNet", {'train_loss_cutout': train_loss, 'train_accuracy_cutout':  train_acc, 'val_accuracy_cutout':  val_acc}, epoch)
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' % (epoch + 1, train_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    tb.close()
    print('Finished Training')

if __name__ == '__main__':
    main()
