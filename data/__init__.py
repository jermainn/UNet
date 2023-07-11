from data.isbi_dataset import isbiDataSet
from torch.utils import data

def CreateTrainDataLoader(args):
    isbi_dataset = isbiDataSet(args.data_dir)

    train_dataset = data.DataLoader(isbi_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

    return train_dataset