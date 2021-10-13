import numpy as np

from models.ner import hyperparameter as hp
from models.ner.preprocess import Preprocess
from models.ner.torchdataset import TorchDataset
from models.ner.model import BERTforNER

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from datetime import datetime

def prepare_training(data_path, gpu_enabled=True):
    p_data = Preprocess(data_path)
    train_set, train_labels, dev_set, dev_labels = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_labels, hp.MAX_LEN)
    dev_dataset = TorchDataset(dev_set, dev_labels, hp.MAX_LEN)

    train_loader = DataLoader(train_dataset, **hp.train_param)
    dev_loader = DataLoader(dev_dataset, **hp.dev_param)

    num_labels = len(p_data.get_classes())
    model = BERTforNER(num_labels)
    if gpu_enabled:
        gpu = hp.GPU
        model.cuda(gpu)

    opti = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    return train_loader, dev_loader, model, opti

def evaluate(model, dataloader, gpu):
    model.eval()

    final_loss = 0

    with torch.no_grad():
        for it, data in enumerate(dataloader):
            seq = data['ids'].cuda(gpu)
            attn_mask = data['mask'].cuda(gpu)
            seg_ids = data['seg_ids'].cuda(gpu)
            labels = data['tags'].cuda(gpu)

            _, loss = model(seq, attn_mask, seg_ids, labels)
            final_loss += loss.item()
    
    return final_loss

def train(model, opti, train_loader, dev_loader, max_ep, gpu):
    hp.save_hp_to_json(datetime.now())
    # best_acc = 0
    best_loss = np.inf
    st = time.time()
    model.train()
    for ep in range(max_ep):
        final_loss = 0
        for it, data in enumerate(train_loader):
            # clear gradients
            opti.zero_grad()

            # convert to cuda tensors
            seq = data['ids'].cuda(gpu)
            attn_mask = data['mask'].cuda(gpu)
            seg_ids = data['seg_ids'].cuda(gpu)
            labels = data['tags'].cuda(gpu)

            # obtaining loss from model
            _, loss = model(seq, attn_mask, seg_ids, labels)

            # backpropagation
            loss.backward()

            # optimisation step
            opti.step()

            final_loss += loss.item()

            if it % 100 == 0:
                print(f"Iteration {it} of epoch {ep} complete. Loss: {loss.item()}; Time taken (s): {time.time() - st}")
                st = time.time()

        dev_loss = evaluate(model, dev_loader, gpu)
        print(f"Epoch {ep} complete. Dev loss: {dev_loss}")
        now_date = datetime.now().strftime("%d%m%Y")
        now_time = datetime.now().strftime("%H%M%S")
        if dev_loss < best_loss:
            print(f"Best dev loss improved from {best_loss} to {dev_loss}")
            best_loss = dev_loss
            print("Saving model..")
            torch.save(
                model.state_dict(),
                f"models/ner/saved_model/BERTforNER_{ep}_{now_date}_{now_time}.dat"
            )

if __name__ == "__main__":
    train_loader, dev_loader, model, opti = prepare_training("data/ner/train.tsv")
    train(model, opti, train_loader, dev_loader, hp.EPOCHS, hp.GPU)