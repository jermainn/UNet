from data import CreateTrainDataLoader
from options.train_options import TrainOptions
import os
from torch import optim
from model import CreateModel
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.timer import Timer
import tensorboardX

def main():
    opt = TrainOptions()
    args = opt.initialize()

    model_name = 'U-Net_ISBI'
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    train_loader = CreateTrainDataLoader(args)
    model, optimizer = CreateModel(args)

    start_epoch = 0
    if args.restore_from is not None:
        start_epoch = int(args.restore_from.rsplit('.', 1)[0].rsplit('/', 1)[1].rsplit('_', 1)[1])

    # print(model)
    # cudnn 加速
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()

    # model.to(device='gpu')
    # 训练过程
    train_writer = tensorboardX.SummaryWriter(
        os.path.join(args.snapshot_dir, "logs", model_name)
    )
    _t = {'iter time' : Timer()}
    _t['iter time'].tic()
    idx = 0
    for epoch in range(start_epoch, args.epochs):
        # 按照 batch_size 开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到 device 中
            image, label = Variable(image).cuda(), Variable(label).cuda()
            # image, label = image.to(device='cuda', dtype=torch.float32), label.to(device='cuda', dtype=torch.float32)
            # image, label = Variable(image), Variable(label)

            # 预测结果
            pred = model(image, lbl=label)
            # 计算loss
            loss = model.loss
            loss.backward()
            optimizer.step()

            train_writer.add_scalar('loss', loss, idx+1)
            print('[items %d][loss %.4f][lr %.6f]'
                  % (idx + 1, loss.item(), optimizer.param_groups[0]['lr']))
            idx += 1

        if not (epoch+1) % args.save_pred_epoch:
            print("taking snapshot ...")
            torch.save(model.state_dict(),
                       os.path.join(args.snapshot_dir, '%s_' %(model_name) + str(epoch+1) + '.pth'))
            _t['iter time'].toc(average=False)
            print('[epoch %d][loss %.4f][lr %.6f][%.2fs]'
                  % (epoch+1, loss.data, optimizer.param_groups[0]['lr'], _t['iter time'].diff))

            _t['iter time'].tic()

    print("The training finished, taking snapshot ...")
    torch.save(model.state_dict(),
               os.path.join(args.snapshot, '%s_' % (model_name) + 'finished' + '.pth'))


if __name__ == "__main__":
    # os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    # os.system('rm tmp')
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))
    # main()
    # print(torch.cuda.is_available())
    # code = str(torch.cuda.current_device())
    # print(code, type(code))
    # os.environ['CUDA_VISIBLE_DEVICES'] = code

    main()


    ## tensorboard 测试
    # from torch.utils.tensorboard import SummaryWriter
    # path = r"E:\我的文件\8 研究生学习\2 语义分割\1 笔记\img"
    # writer = SummaryWriter(log_dir=os.path.join(path, 'mylog'))
    # from PIL import Image
    # img1 = Image.open(os.path.join(path, 'dataVOC.png')).convert('RGB')
    # import numpy as np
    # img1 = np.asarray(img1)
    # img1 = img1.transpose(2, 0, 1)
    # img1 = np.expand_dims(img1, axis=0)
    # print(img1.shape)
    # img1 = torch.Tensor(img1)
    # print(img1.shape)
    # img2 = img1.clone()
    # batch = torch.cat((img1, img2), dim=0)
    # print(batch.shape)
    #
    # import torchvision
    # img_grid = torchvision.utils.make_grid(batch)
    # npimg = img_grid.permute(1, 2, 0).numpy()
    # print(npimg.shape)
    # print(npimg.dtype)
    # npimg = np.uint8(npimg)
    # print(npimg.shape)
    # print(npimg.dtype)
    # # npimg = Image.fromarray(np.uint8(npimg))
    # # print(npimg.dtype)
    # from matplotlib import pyplot as plt
    # plt.imshow(npimg)
    # plt.show()
    #
    # writer.add_image("first", npimg, dataformats='HWC')
    # from matplotlib import pyplot as plt
    #
    # plt.imshow(npimg)
    # plt.show()
    #
    # writer.add_graph()


