import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from models.ner.preprocess import Preprocess

class TorchDataset(Dataset):

    def __init__(self, datafile, labelfile, maxlen) -> None:
        self.sentences = datafile
        self.labels = labelfile

        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        index_sentence = ' '.join(self.sentences[index])

        bert_tokens = self.bert_tokenizer(
            index_sentence,
            padding='max_length',
            truncation=True,
            max_length=self.maxlen
        )

        # extending label list to match the sentence padding
        label = self.labels[index]
        label.extend([2] * self.maxlen)
        label = label[:self.maxlen]

        return {
            'ids': torch.tensor(bert_tokens['input_ids'], dtype=torch.long),
            'mask': torch.tensor(bert_tokens['attention_mask'], dtype=torch.long),
            'seg_ids': torch.tensor(bert_tokens['token_type_ids'], dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # THIS IS DEBUGGING
    p_data = Preprocess('data/ner/train.tsv')
    train_set, train_label, dev_set, dev_label = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_label, 125)
    # print(dataset.__getitem__(1))
    test_data = train_dataset.__getitem__(1)
    print(test_data['ids'])
