from model.unet import UNet
import torch.optim as optim

def CreateModel(args):
    model = UNet(n_channels=1, num_classes=2, bilinear=args.bilinear)

    return model







if __name__ == '__main__':
    model = CreateModel()
    print(model)