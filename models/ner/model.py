import torch
import torch.nn as nn
from transformers import BertModel

class BERTforNER(nn.Module):

    def __init__(self):
        super(BERTforNER, self).__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-cased')

    def forward(self, seq, attn_masks, seg_ids):
        outputs = self.bert_layer(
            seq,
            attention_mask=attn_masks,
            token_type=seg_ids
        )

        return outputs