from model.unet import UNet
import torch.optim as optim
import torch

def CreateModel(args):
    model = UNet(n_channels=args.n_channels, num_classes=args.num_classes, bilinear=args.bilinear, phase=args.set)
    if args.restore_from is not None:
        model.load_state_dict(torch.load(args.restore_from, map_location=lambda storage, loc: storage))
    if args.set == "train":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum)
        optimizer.zero_grad()
        return model, optimizer
    else:
        return model







if __name__ == '__main__':
    model = CreateModel()
    print(model)