from data.isbi_dataset import IsbiDataSet
from data.eval_dataset import EvalDataSet
from torch.utils import data

def CreateTrainDataLoader(args):
    isbi_dataset = IsbiDataSet(args.data_dir)

    train_dataset = data.DataLoader(dataset=isbi_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)

    return train_dataset

def CreateEvalDataLoader(args):
    isbi_dataset = EvalDataSet(args.data_dir)

    evaluation_dataset = data.DataLoader(isbi_dataset,
                                         batch_size=1,
                                         shuffle=False)
    return evaluation_dataset
