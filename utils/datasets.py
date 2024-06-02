import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2


# 数据集类
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.path_file = os.path.join(data_path, 'image/')
        self.image_index = os.listdir(self.path_file)
        self.image_list = [os.path.join(self.path_file, index) for index in self.image_index]

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = image_path.replace('image', 'label')

        # 获得图像
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 将三通道图像转换为单通道图像
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

        # 将label中为255的灰度值转为1，这样是为了方便二分类
        label = label / 255

        # 转换大小格式
        image = image.reshape(1, image.shape[0], image.shape[0])
        label = label.reshape(1, label.shape[0], label.shape[0])
        return image, label

    def __len__(self):
        return len(self.image_list)
