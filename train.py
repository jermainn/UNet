from data import CreateTrainDataLoader
from options.train_options import TrainOptions
import os
from torch import optim
from model import CreateModel
import torch.nn as nn

def main():
    opt = TrainOptions()
    args = opt.initialize()

    model_name = 'U-Net for ISBI'
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    train_loader = CreateTrainDataLoader(args)
    model = CreateModel(args)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    print(model)

    # 训练过程


if __name__ == "__main__":
    main()