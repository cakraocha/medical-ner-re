from models.re import hyperparameter as hp

import matplotlib.pyplot as plt

from collections import OrderedDict

class Preprocess():

    def __init__(self, datafile, split=0.8) -> None:
        # split the dataset into data and labels
        self.sentences = []
        self.labels = []
        # iterate through the file
        # structure of test file and train file are different
        with open(datafile, 'r') as f:
            if datafile == hp.TEST_DATA_DIR:
                next(f)
            for line in f:
                data_line = line.strip().split('\t')
                if data_line[0] != '':
                    if datafile == hp.TRAIN_DATA_DIR:
                        self.sentences.append(data_line[0])
                        self.labels.append(int(data_line[1]))
                    else:
                        self.sentences.append(data_line[1])
                        self.labels.append(int(data_line[2]))

        self.train_split = round(len(self.sentences) * split)
        self.dev_split = len(self.sentences) - self.train_split

        self.train_data = []
        self.train_labels = []
        self.dev_data = []
        self.dev_labels = []

    def get_labels(self):
        return self.labels
    
    def get_sentences(self):
        return self.sentences

    def train_dev_split(self):
        self.train_data = self.sentences[:self.train_split]
        self.train_labels = self.labels[:self.train_split]
        self.dev_data = self.sentences[self.train_split:]
        self.dev_labels = self.labels[self.train_split:]

        return self.train_data, self.train_labels, self.dev_data, self.dev_labels
    
    def count_word_in_sentences(self):
        """
        A function to count number of words in a sentence.
        This function helps to explore the average words of sentences.
        """
        counter = {}
        for data in self.sentences:
            data = data.split()
            if len(data) not in counter:
                counter[len(data)] = 0
            counter[len(data)] += 1
        
        ordered_counter = OrderedDict(sorted(counter.items()))

        return ordered_counter
    
    def plot_word_count(self):
        """
        A function to plot the word counted in sentences
        """
        counter = self.count_word_in_sentences()

        plt.bar(range(len(counter)), list(counter.values()), align='center')
        plt.xticks(range(len(counter)), list(counter.keys()))

        plt.show()

if __name__ == "__main__":
    # THIS IS DEBUGGING
    dataset = Preprocess('data/re/train.tsv')
    dataset.plot_word_count()
