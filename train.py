import torch
from utils.datasets import MyDataset
from torch.utils.data import DataLoader
from model.network import Unet
from torch import optim
import torch.nn as nn

if __name__=="__main__":

    # 初始化参数
    device = "cuda:0"

    # 构造数据集
    data = MyDataset('./data/train/')
    train_loader = DataLoader(data, batch_size=2, shuffle=True)

    # 定义模型
    model = Unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 500
    for epoch in range(epochs):
        model.train()
        loss_ = 0
        for i, (image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 前向传播
            predict = model(image)

            # 计算损失
            loss = criterion(predict, label)
            loss_ = loss_ + loss

            # 更新参数
            loss.backward()
            optimizer.step()
        print("epoch is {} and loss is {}".format(epoch, loss_ / 30))

        # 保存模型
        if epoch % 20 == 0:
            print('saving the model at the end of epoch {}'.format(epoch))
            torch.save(model.state_dict(), './result/models/model_{}.pth'.format(epoch))


