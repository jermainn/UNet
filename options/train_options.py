import argparse
import os.path as osp

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="U-Net Model")
        parser.add_argument("--data-dir", type=str, default='./dataset', help="Path to directory containing the train and test dataset")
        parser.add_argument("--batch-size", type=int, default=1, help="imput batch size.")
        parser.add_argument("--learning-rate", type=float, default=0.0001, help="initial learning rate.")
        parser.add_argument("--weight-decay", type=float, default=1e-8, help="Regularisation parameter for loss.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")

        parser.add_argument("--bilinear", type=bool, default=False, help="Use bilinear upsampling or not.")
        parser.add_argument("--set", type=str, default='train', help="train or not.")
        parser.add_argument("--snapshot-dir", type=str, default='./snapshots', help="Where to save snapshots of the model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")

        # parser.add_argument()

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            # {:>25} 表示 25 个字符，右对齐
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to file
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
