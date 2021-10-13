import torch
import torch.nn as nn
from transformers import BertModel

class BERTforNER(nn.Module):

    def __init__(self, num_labels):
        super(BERTforNER, self).__init__()

        self.num_labels = num_labels

        self.bert_layer = BertModel.from_pretrained('bert-base-cased', return_dict=False)
        self.bert_drop_1 = nn.Dropout(0.2)
        self.out = nn.Linear(768, num_labels)

    def calculate_loss(self, output, target, mask, num_labels):
        criterion = nn.CrossEntropyLoss()
        # below code is to calculate only the ones with mask 1
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, num_labels)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(criterion.ignore_index).type_as(target)
        )
        loss = criterion(active_logits, active_labels)

        return loss

    def forward(self, seq, attn_masks, seg_ids, labels):
        o1, _ = self.bert_layer(
            seq,
            attention_mask=attn_masks,
            token_type_ids=seg_ids
        )  # we're taking the sequence, that is the o1
        bo_drop = self.bert_drop_1(o1)
        output = self.out(bo_drop)

        loss = self.calculate_loss(output, labels, attn_masks, self.num_labels)

        return output, loss
