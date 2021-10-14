import numpy as np

from models.re import hyperparameter as hp
from models.re.preprocess import Preprocess
from models.re.torchdataset import TorchDataset
from models.re.model import BERTforNER, BIOBERTforNER

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

import time
from datetime import datetime

def prepare_training(data_path, use_biobert=False, gpu_enabled=True):
    p_data = Preprocess(data_path)
    train_set, train_labels, dev_set, dev_labels = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_labels, hp.MAX_LEN, use_biobert=use_biobert)
    dev_dataset = TorchDataset(dev_set, dev_labels, hp.MAX_LEN, use_biobert=use_biobert)

    train_loader = DataLoader(train_dataset, **hp.train_param)
    dev_loader = DataLoader(dev_dataset, **hp.dev_param)

    if use_biobert:
        model = BIOBERTforNER(
            hp.BIOBERT_CONFIG,
            hp.BIOBERT_MODEL,
            hp.DEVICE,
        )
    else:
        model = BERTforNER()
    if gpu_enabled:
        model.cuda(hp.GPU)

    opti = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    return train_loader, dev_loader, model, opti

def get_acc_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()

    return acc

def evaluate(model, dataloader, gpu):
    model.eval()

    preds = []
    labels = []
    final_loss = 0

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            for k, v in data.items():
                data[k] = v.cuda(gpu)

            logits, loss = model(**data)

            probs = torch.sigmoid(logits.unsqueeze(-1))
            soft_probs = (probs > 0.5).long()
            for prob in soft_probs:
                preds.append(prob.squeeze().item())
            for label in data['labels']:
                labels.append(label.float().item())

            final_loss += loss.item()

    return f1_score(labels, preds), final_loss 

def train(model, opti, train_loader, dev_loader, max_ep, gpu):
    hp.save_hp_to_json(datetime.now())
    best_loss = np.inf
    best_f1 = 0
    st = time.time()
    model.train()
    for ep in range(max_ep):
        final_loss = 0
        for it, data in enumerate(train_loader):
            # clear gradients
            opti.zero_grad()

            # convert to cuda tensors
            for k, v in data.items():
                data[k] = v.cuda(gpu)

            # obtaining loss from model
            logits, loss = model(**data)

            # backpropagation
            loss.backward()

            # optimisation step
            opti.step()

            final_loss += loss.item()

            if it % 100 == 0:
                acc = get_acc_from_logits(logits, data['labels'])
                print(f"Iteration {it} of epoch {ep} complete. Acc: {acc}; Loss: {loss.item()}; Time taken (s): {time.time() - st}")
                st = time.time()

        dev_f1, dev_loss = evaluate(model, dev_loader, gpu)
        print(f"Epoch {ep} complete. Dev F1: {dev_f1}; Dev loss: {dev_loss}")
        now_date = datetime.now().strftime("%Y%m%d")
        now_time = datetime.now().strftime("%H%M%S")
        if dev_f1 > best_f1:
            print(f"Best f1 score improved from {best_f1} to {dev_f1}")
            best_f1 = dev_f1
            print("Saving model..")
            torch.save(
                model.state_dict(),
                f"{hp.SAVED_MODEL_DIR}/BERTforNER_{ep}_{now_date}_{now_time}.dat"
            )

if __name__ == "__main__":
    train_loader, dev_loader, model, opti = prepare_training(hp.TRAIN_DATA_DIR, use_biobert=True)
    train(model, opti, train_loader, dev_loader, hp.EPOCHS, hp.GPU)