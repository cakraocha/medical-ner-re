import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer, AutoTokenizer

from models.ner.preprocess import Preprocess

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
        tags = self.labels[index]

        ids = []
        target_tag = []

        for i, s in enumerate(sentence):
            if self.use_biobert:
                inputs = self.biobert_tokenizer.encode(
                    s,
                    add_special_tokens=False
                )
            else:
                inputs = self.bert_tokenizer.encode(
                    s,
                    add_special_tokens=False
                )
            # we want to make all of inputs the same length
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        # we do minus 2 because we need to add special tokens
        ids = ids[:self.maxlen - 2]
        target_tag = target_tag[:self.maxlen - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.maxlen - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        # extending label list to match the sentence padding
        # label = self.labels[index]
        # label.extend([2] * self.maxlen)
        # label = label[:self.maxlen]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'seg_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(target_tag, dtype=torch.long)
        }

if __name__ == "__main__":
    # THIS IS DEBUGGING
    p_data = Preprocess('data/ner/train.tsv')
    train_set, train_label, dev_set, dev_label = p_data.train_dev_split()
    train_dataset = TorchDataset(train_set, train_label, 125, use_biobert=True)
    # print(dataset.__getitem__(1))
    test_data = train_dataset.__getitem__(1)
    print(test_data['ids'])
