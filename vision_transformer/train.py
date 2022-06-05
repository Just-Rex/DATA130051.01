import os
import math
import argparse
import configparser
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import vit_base
from utils import read_split_data, train_one_epoch, evaluate


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['TRAIN'] # 选择对应参数
    log_dir = params['log_dir'] # tensorboard存放位置
    model_path = params['model_path'] # 训练文件存放路径
    image_path = params['image_path'] # 训练图像存放路径
    batch_size = int(params['batch_size']) # 批大小
    num_classes = int(params['num_classes']) # 训练图像类别总数
    patch_size = int(params['patch_size']) # 设置patch尺寸
    num_heads = int(params['num_heads']) # 设置头个数
    mlp_hidden_dim = int(params['mlp_hidden_dim']) # 设置mlp隐藏层数量
    depth = int(params['depth']) # 设置头个数
    learning_rate = float(params['learning_rate']) # 优化器学习率
    num_epochs = int(params['num_epochs']) # 训练轮数
    # 使用tensorboard可视化训练过程
    tb = SummaryWriter(log_dir=log_dir)

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 设置模型参数存放位置
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 图像预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 使用线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集
    train_dataset = datasets.CIFAR100(root = image_path, train = True, download = True, transform = data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = nw)

    # 加载验证集
    val_dataset = datasets.CIFAR100(root = image_path, train = False, download = True, transform = data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = nw)

    # 实例化模型
    model = vit_base(num_classes=num_classes, patch_size=patch_size, num_heads=num_heads, depth=depth, mlp_hidden_dim=mlp_hidden_dim).to(device)

    # 加载权重
    # model.load_state_dict(torch.load(os.path.join(model_path, 'model-best.pth')))
    # print("Successfully loading")

    # 输出模型总参数
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    # 设置优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr = learning_rate, momentum=0.9, weight_decay=5E-5)

    # 设置学习率更新策略
    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    for epoch in range(1, num_epochs+1):
        # 训练过程
        train_loss, train_acc = train_one_epoch(model=model,
                              optimizer=optimizer,
                              data_loader=train_loader,
                              device=device,
                              epoch=epoch)
        scheduler.step()

        # 验证过程
        val_loss, val_acc = evaluate(model=model,
                        data_loader=val_loader,
                        device=device,
                        epoch=epoch)

        tags = ["train_loss_baseline", "train_acc_baseline", "val_loss_baseline", "val_acc_baseline", "learning_rate_baseline"]
        tb.add_scalars("Transformer", {tags[0]: train_loss, tags[1]: train_acc, tags[2]: val_loss, tags[3]: val_acc, tags[4]: optimizer.param_groups[0]["lr"]}, epoch)
        # tb.add_scalar(tags[0], train_loss, epoch)
        # tb.add_scalar(tags[1], train_acc, epoch)
        # tb.add_scalar(tags[2], val_loss, epoch)
        # tb.add_scalar(tags[3], val_acc, epoch)
        # tb.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_path, 'model-best.pth'))

    tb.close()  
    print('Training is over, the best val_acc is {}' % (best_acc))

if __name__ == '__main__':
    main()
