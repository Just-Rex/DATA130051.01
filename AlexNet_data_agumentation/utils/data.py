import random
import numpy as np
import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def mixup_data(images, labels, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    mixup_images = torch.zeros([images.shape[0], 3, 224, 224])
    mixup_labels = torch.zeros([images.shape[0], 2]).type(torch.LongTensor)

    for i in range(images.shape[0]-1):
        r_dex = random.randint(0,images.shape[0]-1)
        if i % 3 == 0:
            mixup_images[i] = lam * images[i] + (1 - lam) * images[r_dex]
            mixup_labels[i][0], mixup_labels[i][1] = labels[i], labels[r_dex]
        else:
            mixup_images[i] = images[i]
            mixup_labels[i][0], mixup_labels[i][1] = labels[i], labels[i]
    return mixup_images, mixup_labels, lam


def mixup_loss(criterion, outputs, labels, lam):
    return lam * criterion(outputs, labels[:, 0]) + (1 - lam) * criterion(outputs, labels[:, 1])


def cutmix_data(images, labels, alpha):
    lam = np.random.beta(alpha, alpha)
    cutmix_images = torch.ones([images.shape[0], 3, 224, 224])
    cutmix_labels = torch.zeros([images.shape[0], 2]).type(torch.LongTensor)
    x1, y1, x2, y2 = rand_bbox(images.shape, lam)

    for i in range(images.shape[0]-1):
        r_dex = random.randint(0, images.shape[0]-1)
        x0 = random.randint(x2-223, x1)
        y0 = random.randint(y2-223, y1)
        cutmix_images[i][:,x1-x0:x2-x0, y1-y0:y2-y0] = 0
        cutmix_images[i] = cutmix_images[i] * images[i]
        if i % 3 == 0:
            cutmix_images[i][:,x1-x0:x2-x0, y1-y0:y2-y0] = images[r_dex][:,x1-x0:x2-x0, y1-y0:y2-y0]
            cutmix_labels[i][0], cutmix_labels[i][1] = labels[i], labels[r_dex]
        else:
            cutmix_images[i][:,x1-x0:x2-x0, y1-y0:y2-y0] = images[i][:,x1-x0:x2-x0, y1-y0:y2-y0]
            cutmix_labels[i][0], cutmix_labels[i][1] = labels[i], labels[i]
    return cutmix_images, cutmix_labels, lam

def cutmix_loss(criterion, outputs, labels, lam):
    return lam * criterion(outputs, labels[:, 0]) + (1 - lam) * criterion(outputs, labels[:, 1])


def cutout_data(images, labels, alpha):
    lam = np.random.beta(alpha, alpha)
    cutout_images = torch.ones([images.shape[0], 3, 224, 224])

    for i in range(images.shape[0]-1):
        x1, y1, x2, y2 = rand_bbox(images.shape, lam)
        if i % 3 == 0:
            cutout_images[i][:,x1:x2, y1:y2] = 0
        cutout_images[i] = cutout_images[i] * images[i]
    cutout_labels = labels
    return cutout_images, cutout_labels