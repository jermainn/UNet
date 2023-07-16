# UNet

( 进行中…… )

基于 Pytorch 框架实现 UNet 网络

论文地址：[arxiv.org/pdf/1505.04597.pdf](https://arxiv.org/pdf/1505.04597.pdf) 

笔记：

​	GitHub: [2 UNet 模型 (jermainn.github.io)](https://jermainn.github.io/docsNote/#/Semantic_Segmentation/2_UNet_model)

​	Gitee: [2 UNet 模型 (gitee.io)](https://jermainn.gitee.io/docsnote/#/Semantic_Segmentation/2_UNet_model)

### 模型调整

<img src="https://cdn.jsdelivr.net/gh/jermainn/imgpic@master/note_img/UNet.webp" alt="UNet" style="zoom:67%;" />

模型的实现使用 `Pytorch` 框架

在我的模型的实现过程中，在论文中所述模型的基础上，在卷积的过程中引入了 `padding` ，而没有使用 `overlap-tile` 策略

另外，提供参数 `bilinear` ，可以选择是否使用双线性插值的上采样 `nn.Upsample` 来替换原论文汇中的转置卷积 `nn.ConvTranspose2d` 

### 进度

已经完成模型的构建和数据读取和模型训练部分程序设计，下一步进行模型训练和模型评估

进行中……
