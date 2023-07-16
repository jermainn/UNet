from torch.utils import data
import glob
import os
from PIL import Image
import random
import numpy as np

class EvalDataSet(data.Dataset):
    def __init__(self, root):
        """
        类初始化函数：
        根据指定的路径读取所有图片数据
        """
        self.root = root
        self.is_label = is_label
        # # 返回不带后缀的文件名
        self.img_ids = [file.split('.')[0]
                        for file in os.listdir(os.path.join(root, "test/image"))]
        # print(self.img_ids, len(self.img_ids))

        # # 返回所有满足要求的文件路径列表
        # self.imgs_path = glob.glob(os.path.join(data_root, "train/image/*.png"))
        # print(self.imgs_path)

    def __len__(self):
        """
        返回数据量多少
        """
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        数据获取函数：
        数据的读取与预处理
        """
        name = self.img_ids[index]
        # 图片和标签
        image = Image.open(
            os.path.join(self.root, "train/image/%s.png" % name))

        # 读入的单通道图像 否则用 .convert('L')
        # resize
        image = image.resize(image.size, Image.BICUBIC)
        image = np.asarray(image, np.float32)
        # 由于 Image.open 读入的灰度图只有单个通道，所以需要增加一个通道
        # N C H W
        image = np.expand_dims(image, axis=0)

        # if self.is_label:
        #     label = Image.open(os.path.join(self.root, "train/label/%s.png" % name))
        #     label = label.resize(image.size, Image.BICUBIC)
        #     label = np.asarray(label, np.float32)
        #
        #     if label.max() > 1:
        #         label = label / 255      # 将标签 255 变为 1
        #     # 由于 Image.open 读入的灰度图只有单个通道，所以需要增加一个通道
        #     # N C H W
        #     image, label = np.expand_dims(image, axis=0), np.expand_dims(label, axis=0)
        #
        #     return image.copy(), label.copy(), name
        return image.copy(), name


if __name__ == "__main__":
    # data_path = "./../dataset"
    # dataset = isbiDataSet(data_path)
    # lbl = Image.open("./../dataset/train/label/0.png")
    # print(lbl.size)
    # import numpy as np
    # lblarr = np.asarray(lbl)
    # print(np.unique(lblarr))
    # print(lblarr.shape)
    # lblarr = lblarr/255
    # print(np.unique(lblarr))

    root = "./../dataset"
    dataset = IsbiDataSet(root)
    print("数据的个数：", len(dataset))
    train_loader = data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for image, label in train_loader:
        print(image.shape, image.size())


