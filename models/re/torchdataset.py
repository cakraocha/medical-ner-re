import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

from models.re.preprocess import Preprocess

class TorchDataset(Dataset):

    def __init__(self, datafile, labelfile, maxlen, use_biobert=False) -> None:
        self.sentences = datafile
        self.labels = labelfile

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-cased',
            do_lower_case=False
        )
        self.biobert_tokenizer = BertTokenizer(
            vocab_file='biobert_v1.1_pubmed/vocab.txt',
            do_lower_case=False
        )

        self.use_biobert = use_biobert
        self.maxlen = maxlen

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]

        if self.use_biobert:
            bert_tokens = self.biobert_tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=self.maxlen
            )
        else:
            bert_tokens = self.bert_tokenizer(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=self.maxlen
            )

        return {
            'ids': torch.tensor(bert_tokens['input_ids'], dtype=torch.long),
            'mask': torch.tensor(bert_tokens['attention_mask'], dtype=torch.long),
            'seg_ids': torch.tensor(bert_tokens['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # THIS IS DEBUGGING
    p_data = Preprocess('data/re/train.tsv')
    train_set, train_label, dev_set, dev_label = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_label, 125, use_biobert=True)
    test_data = train_dataset.__getitem__(1)
    print(test_data['ids'])
