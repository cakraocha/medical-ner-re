import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import shuffle

from collections import OrderedDict

class Preprocess():

    def __init__(self, datafile, split=0.8) -> None:
        # split the dataset into data and labels
        self.sentences = []
        self.raw_labels = []
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
                    self.raw_labels.append(label)
                    sentence = []
                    label = []

        self.le = preprocessing.LabelEncoder()
        label_list = [x for l in self.raw_labels for x in l]
        self.le.fit(label_list)

        self.labels = []
        for l in self.raw_labels:
            self.labels.append(self.le.transform(l).tolist())

        self.train_split = round(len(self.sentences) * split)
        self.dev_split = len(self.sentences) - self.train_split

        self.train_sentences = []
        self.train_labels = []
        self.dev_sentences = []
        self.dev_labels = []

    def get_labels(self):
        return self.labels
    
    def get_sentences(self):
        return self.sentences

    def get_classes(self):
        return self.le.classes_
    
    def train_dev_split(self):
        sentences_shuffled, labels_shuffled = shuffle(self.sentences, self.labels)
        self.train_sentences = sentences_shuffled[:self.train_split]
        self.train_labels = labels_shuffled[:self.train_split]
        self.dev_sentences = sentences_shuffled[self.train_split:]
        self.dev_labels = labels_shuffled[self.train_split:]

        return self.train_sentences, self.train_labels, self.dev_sentences, self.dev_labels
    
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
    dataset = Preprocess('data/ner/train.tsv')
    # print(dataset.data_to_dict())
    # print(dataset.get_labels())
    # print(dataset.get_classes())
    # dataset.plot()
