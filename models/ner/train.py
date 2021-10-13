from models.ner import hyperparameter as hp
from models.ner.preprocess import Preprocess
from models.ner.torchdataset import TorchDataset
from models.ner.model import BERTforNER

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import time

def prepare_training(data_path, gpu_enabled=True):
    p_data = Preprocess(data_path)
    train_set, train_labels, dev_set, dev_labels = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_labels, hp.MAX_LEN)
    dev_dataset = TorchDataset(dev_set, dev_labels, hp.MAX_LEN)

    train_loader = DataLoader(train_dataset, **hp.train_param)
    dev_loader = DataLoader(dev_dataset, **hp.dev_param)
    
    model = BERTforNER(hp.MAX_LEN)
    if gpu_enabled:
        gpu = 0
        model.cuda(gpu)

    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    return train_loader, dev_loader, model, criterion, opti

def get_acc_from_output(output, labels):
    probs = torch.sigmoid(output.unsqueeze(-1))
    soft_probs = (max(probs)).long()
    acc = (soft_probs.squeeze() == labels).float().mean()

    return acc

def evaluate(model, criterion, dataloader, gpu):
    model.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for data in dataloader:
            seq, attn_mask, labels = data['ids'].cuda(gpu), data['mask'].cuda(gpu), data['tags'].cuda(gpu)
            output = model(seq, attn_mask)
            mean_loss += criterion(output.squeeze(-1), labels.float()).item()
            mean_acc += get_acc_from_output(output, labels)
            count += 1
    
    return mean_acc / count, mean_loss / count

def train(model, criterion, opti, train_loader, dev_loader, max_ep, gpu):
    best_acc = 0
    st = time.time()
    model.train()
    for ep in range(max_ep):
        for it, data in enumerate(train_loader):
            # clear gradients
            opti.zero_grad()

            # convert to cuda tensors
            seq, attn_mask, labels = data['ids'].cuda(gpu), data['mask'].cuda(gpu), data['tags'].cuda(gpu)

            # obtaining output from model
            output = model(seq, attn_mask)

            # compute loss
            loss = criterion(output.squeeze(-1), labels.float())

            # backpropagation
            loss.backward()

            # optimisation step
            opti.step()

            if it % 100 == 0:
                acc = get_acc_from_output(output, labels)
                print(f"Iteration {it} of epoch {ep} complete. \
                    Loss: {loss.item()}; Acc: {acc}; \
                    Time taken (s): {time.time() - st}")
                st = time.time()

        dev_acc, dev_loss = evaluate(model, criterion, dev_loader, gpu)
        print(f"Epoch {ep} complete. Dev acc: {dev_acc}; Dev loss: {dev_loss}")
        if dev_acc > best_acc:
            print(f"Best dev acc improved from {best_acc} to {dev_acc}")
            best_acc = dev_acc
            print("Saving model..")
            torch.save(model.state_dict(), f"models/ner/saved_model/BERTforNER_{ep}.dat")

if __name__ == "__main__":
    train_loader, dev_loader, model, criterion, opti = prepare_training("data/ner/train.tsv")