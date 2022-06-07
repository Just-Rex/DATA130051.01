from mnist import MNIST
from utils import DataLoader

def load_data_mnist(image_path='./data', batch_size=64, shuffle=True):
    mndata = MNIST(image_path)
    train_iter = DataLoader(mndata.load_training(), batch_size, shuffle)
    test_iter = DataLoader(mndata.load_testing(), batch_size)
    print(f'Loading MNIST dataset. {len(train_iter)} items for training, {len(test_iter)} items for testing. batch_size={batch_size}')
    print(f'X size: {train_iter.images.shape[1]}, MNIST image scale 28x28, total_cls=10.')
    return train_iter, test_iter, train_iter.images.shape[1]