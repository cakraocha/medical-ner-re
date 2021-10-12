import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from collections import OrderedDict

class Preprocess(Dataset):

    def __init__(self, datafile, maxlen) -> None:
        # split the dataset into data and labels
        self.sentences = []
        self.labels = []
        sentence = []
        label = []
        # dataset contains spaces between sentences
        with open(datafile, 'r') as f:
            for line in f:
                data_line = line.strip().split('\t')
                if data_line[0] != '':
                    sentence.append(data_line[0])
                    label.append(data_line[1])
                else:
                    self.sentences.append(sentence)
                    self.labels.append(label)
                    sentence = []
                    label = []

        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.maxlen = maxlen

    def __getitem__(self, index):
        index_sentence = ' '.join(self.sentences[index])

        bert_tokens = self.bert_tokenizer(
            index_sentence,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen
        )

        token_ids = torch.tensor(bert_tokens['input_ids'])
        attn_mask = torch.tensor(bert_tokens['attention_mask'])
        seg_ids = torch.tensor(bert_tokens['token_type_ids'])

        label = self.labels[index]

        return token_ids, attn_mask, seg_ids, label
    
    def data_to_dict(self):
        counter = {}
        for data in self.sentences:
            if len(data) not in counter:
                counter[len(data)] = 0
            counter[len(data)] += 1
        
        ordered_counter = OrderedDict(sorted(counter.items()))

        return ordered_counter
    
    def plot(self):
        counter = self.data_to_dict()

        plt.bar(range(len(counter)), list(counter.values()), align='center')
        plt.xticks(range(len(counter)), list(counter.keys()))

        plt.show()

if __name__ == "__main__":
    # THIS IS DEBUGGING
    dataset = Preprocess('data/ner/train.tsv', 100)
    # print(dataset.__getitem__(2))
    print(dataset.data_to_dict())
    # dataset.plot()
