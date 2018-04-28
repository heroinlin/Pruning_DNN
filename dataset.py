import torch.utils.data as data
from torchvision import transforms, datasets


class CDataset(data.Dataset):
    def __init__(self, phase='train'):
        super(CDataset, self).__init__()
        self._phase = phase
        self._dataset_path = r"F:\Database\cifar-10\cifar-10-python"
        transform_train = transforms.Compose(
            [# 随机截取
             transforms.RandomCrop(32, padding=4),
             # 图像翻转
             transforms.RandomHorizontalFlip(),
             # 数据张量化 (0,255) >> (0,1)
             transforms.ToTensor(),
             # 数据正态分布 (0,1） >> (-1,1)
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        # 加载数据靠 train 做以区分 训练集和测试集
        if self._phase == 'train':
            self.dataset = datasets.CIFAR10(self._dataset_path, train=True, transform=transform_train)
        elif self._phase == 'validation':
            self.dataset = datasets.CIFAR10(self._dataset_path, train=False, transform=transform_test)
