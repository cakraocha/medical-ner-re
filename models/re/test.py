from models.re import hyperparameter as hp
from models.re.preprocess import Preprocess
from models.re.torchdataset import TorchDataset
from models.re.model import BERTforNER, BIOBERTforNER

import torch

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predict(datapath, modelpath, use_biobert=False):
    p_data = Preprocess(datapath, split=1)
    test_set = p_data.get_sentences()
    test_labels = p_data.get_labels()
    test_dataset = TorchDataset(test_set, test_labels, hp.MAX_LEN, use_biobert=use_biobert)

    device = torch.device("cuda")
    if use_biobert:
        model = BIOBERTforNER(
            hp.BIOBERT_CONFIG,
            hp.BIOBERT_MODEL,
            device
        )
    else:
        model = BERTforNER()
    model.load_state_dict(torch.load(modelpath))
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for _, data in enumerate(test_dataset):
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)

            tag, _ = model(**data)

            probs = torch.sigmoid(tag.unsqueeze(-1))
            soft_probs = (probs > 0.5).long()

            preds.append(soft_probs.squeeze().item())
    
    return preds, test_labels

if __name__ == "__main__":
    datapath = 'data/re/test.tsv'
    modelpath = 'models/re/saved_model/BERTforNER_2_20211015_000759.dat'
    preds, test_labels = predict(datapath, modelpath, use_biobert=True)
    print(f1_score(test_labels, preds, average='weighted'))
    print(accuracy_score(test_labels, preds))
    print(confusion_matrix(test_labels, preds, labels=[0, 1]))
