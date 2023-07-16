from options.test_options import TestOptions
import os
from model import CreateModel
from data import EvalDataSet
from torch.autograd import Variable

def main():
    opt = TestOptions()
    args = opt.initialize()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = CreateModel(args)

    model.eval()
    model.cuda()

    eval_loader = EvalDataSet(args)

    for index, batch in enumerate(eval_loader):
        image, name = batch
        print(image.shape, name)
        output = model(Variable(image).cuda())



