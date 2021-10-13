import torch
import torch.nn as nn
from transformers import BertModel

class BERTforNER(nn.Module):

    def __init__(self, out):
        super(BERTforNER, self).__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(768, out)

    def forward(self, seq, attn_masks):
        bert_output = self.bert_layer(
            seq,
            attention_mask=attn_masks
        )
        dropout = self.dropout(bert_output[0])
        output = self.linear(dropout)

        return output