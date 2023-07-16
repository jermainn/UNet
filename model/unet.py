import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    """
    每次下采样之后进行两次卷积
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.maxpool(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            # 双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 保证 x1 与 x2 同大小，可以拼接
        # [left, right, top, bottom] 填充行列数
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # [N, C, H, W] 在 channel 上拼接
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    """
    这里父级采用了，nn.Sequential
    """
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

# class OutConv(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(OutConv, self).__init__()
#         self.out = nn.Sequential(
#             nn.Conv2d(in_channels, num_classes, kernel_size=1)
#         )
#     def forward(self, x):
#         return self.out(x)


class UNet(nn.Module):
    def __init__(self,
                 n_channels,
                 num_classes,
                 bilinear=False,
                 base_c=64,
                 phase='train'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.phase = phase

        # the contracting path
        self.in_conv = DoubleConv(self.n_channels, base_c)
        self.down1 = DownSampling(base_c, base_c*2)
        self.down2 = DownSampling(base_c*2, base_c*4)
        self.down3 = DownSampling(base_c*4, base_c*8)
        # 转置卷积 和 双线性插值 在每块的最后一层不太一样
        factor = 2 if bilinear else 1
        self.down4 = DownSampling(base_c*8, base_c*16//factor)
        # the expansive path
        self.up1 = UpSampling(base_c*16, base_c*8//factor, bilinear)
        self.up2 = UpSampling(base_c*8, base_c*4//factor, bilinear)
        self.up3 = UpSampling(base_c*4, base_c*2//factor, bilinear)
        self.up4 = UpSampling(base_c*2, base_c, bilinear)
        # out
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x, lbl=None):
        # the contracting path
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        # the expansive path
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if self.phase == 'train' and lbl is not None:
            self.loss = self.criterion_loss(x, lbl)

        return x


    def criterion_loss(self, pred, lbl):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(pred, lbl)


if __name__ == '__main__':
    # print(torch.cuda.get_arch_list())
    # exit()
    net = UNet(n_channels=1, num_classes=1)
    net.cuda()
    from torchsummary import summary
    summary(net, (1, 320, 320), batch_size=1)
    print(net)
    # torch.cuda.is

    # from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(r"E:\我的文件\8 研究生学习\2 语义分割\1 笔记\img\mylog")

    from PIL import Image
    import numpy as np
    img = Image.open(r"E:\我的文件\8 研究生学习\2 语义分割\1 笔记\img\dataVOC.png")
    img = np.asarray(img)
    writer.add_image('first', img, dataformats='HWC')
    writer.add_graph(net, input_to_model=torch.rand(size=(1, 1, 320, 320)).cuda())


