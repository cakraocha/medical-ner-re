from torch.utils.data.dataloader import DataLoader
from models.ner import hyperparameter as hp
from models.ner.preprocess import Preprocess
from models.ner.torchdataset import TorchDataset
from models.ner.model import BERTforNER

from torch.utils.data import DataLoader

def train():
    p_data = Preprocess('data/ner/train.tsv')
    train_set, train_labels, dev_set, dev_labels = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_labels, hp.MAX_LEN)
    dev_dataset = TorchDataset(dev_set, dev_labels, hp.MAX_LEN)

    train_loader = DataLoader(train_dataset, **hp.train_param)
    dev_loader = DataLoader(dev_dataset, **hp.dev_param)

    model = BERTforNER(hp.MAX_LEN)
    