import torch
from utils.datasets import MyDataset
from torch.utils.data import DataLoader
from model.network import Unet
from torch import optim
import torch.nn as nn
import os
import cv2
import numpy as np

if __name__ == "__main__":
    device = "cuda:0"
    model = Unet()
    model.to(device)
    model.load_state_dict(torch.load('./result/models/model_480.pth'))

    image_index = os.listdir('./data/test/')
    image_list = [os.path.join('./data/test/', index) for index in image_index]

    for image_path in image_list:
        save_res_path = '.'+image_path.split('.')[1] + '_res.png'
        # 读取图像
        img = cv2.imread(image_path)
        # 转换为灰度图
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device, dtype=torch.float32)
        pred = model(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred>=0.5] = 255
        pred[pred<0.5] = 0
        cv2.imwrite(save_res_path,pred)
        print('{} saved'.format(save_res_path))
