from models.ner import hyperparameter as hp
from models.ner.preprocess import Preprocess
from models.ner.torchdataset import TorchDataset
from models.ner.model import BERTforNER, BIOBERTforNER

import torch

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predict(datapath, modelpath, use_biobert=False):
    p_data = Preprocess(datapath, split=1)
    test_set = p_data.get_sentences()
    test_labels = p_data.get_labels()
    test_dataset = TorchDataset(test_set, test_labels, hp.MAX_LEN)

    num_labels = len(p_data.get_classes())
    device = torch.device("cuda")
    if use_biobert:
        model = BIOBERTforNER(
            hp.BIOBERT_CONFIG,
            hp.BIOBERT_MODEL,
            device,
            num_labels
        )
    else:
        model = BERTforNER(num_labels)
    model.load_state_dict(torch.load(modelpath))
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            # seq = data['ids'].to(device).unsqueeze(0)
            # attn_mask = data['mask'].to(device).unsqueeze(0)
            # seg_ids = data['seg_ids'].to(device).unsqueeze(0)
            # labels = data['tags'].to(device).unsqueeze(0)

            tag, _ = model(**data)

            tag = tag.argmax(2).cpu().numpy().reshape(-1)
            tag = tag[:len(test_labels[idx])]
            # tag = p_data.le.inverse_transform(tag)

            # print(tag.tolist())
            # print(test_labels[idx])
            # print(test_set[idx])

            preds.append(tag.tolist())

            # if idx == 3:
            #     break

    
    return preds, test_labels

if __name__ == "__main__":
    datapath = 'data/ner/test.tsv'
    modelpath = 'models/ner/saved_model/BERTforNER_0_20211014_201754.dat'
    preds, test_labels = predict(datapath, modelpath, use_biobert=True)
    preds = [p for l in preds for p in l]
    test_labels = [tl for l in test_labels for tl in l]
    # print(len(preds))
    # print(len(test_labels))
    print(f1_score(test_labels, preds, average=None))
    print(accuracy_score(test_labels, preds))
    print(confusion_matrix(test_labels, preds, labels=[0, 1, 2]))
