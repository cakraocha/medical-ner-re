from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

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

    def forward(self, ids, mask, seg_ids, labels):
        o1, _ = self.bert_layer(
            ids,
            attention_mask=mask,
            token_type_ids=seg_ids
        )  # we're taking the sequence, that is the o1
        bo_drop = self.bert_drop_1(o1)
        output = self.out(bo_drop)

        loss = self.calculate_loss(output, labels, mask, self.num_labels)

        return output, loss

class BIOBERTforNER(nn.Module):

    def __init__(self, configpath, statepath, device, num_labels):
        super(BIOBERTforNER, self).__init__()

        config = BertConfig.from_json_file(configpath)
        temp_state = torch.load(statepath, map_location=device)
        state_dict = OrderedDict()
        for i in list(temp_state.keys())[:199]:
            x = i
            if i.find('bert') > -1:
                x = '.'.join(i.split('.')[1:])
            state_dict[x] = temp_state[i]

        self.biobert_layer = BertModel(config)
        self.biobert_layer.load_state_dict(state_dict, strict=False)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(self.biobert_layer.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.num_labels = num_labels

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

    def forward(self, ids, mask, seg_ids, labels):
        o1 = self.biobert_layer(
            ids,
            attention_mask=mask
        )
        o1 = o1[0]
        # print(o1)
        out = self.dropout(o1)
        out = self.output(out)

        loss = self.calculate_loss(out, labels, mask, self.num_labels)

        return out, loss

if __name__ == "__main__":
    pass
