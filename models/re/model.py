from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTforNER(nn.Module):

    def __init__(self):
        super(BERTforNER, self).__init__()

        self.bert_layer = BertModel.from_pretrained('bert-base-cased')
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, seg_ids, labels):
        o1 = self.bert_layer(
            ids,
            attention_mask=mask,
            token_type_ids=seg_ids
        )
        cont_reps = o1.last_hidden_state
        cls_rep = cont_reps[:, 0]  # taking the cls
        bo_drop = self.bert_drop_1(cls_rep)
        output = self.out(bo_drop)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output.squeeze(-1), labels.float())

        return output, loss

class BIOBERTforNER(nn.Module):

    def __init__(self, configpath, statepath, device):
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
        self.output = nn.Linear(self.biobert_layer.config.hidden_size, 1)

    # def calculate_loss(self, output, target, mask, num_labels):
    #     criterion = nn.CrossEntropyLoss()
    #     # below code is to calculate only the ones with mask 1
    #     active_loss = mask.view(-1) == 1
    #     active_logits = output.view(-1, num_labels)
    #     active_labels = torch.where(
    #         active_loss,
    #         target.view(-1),
    #         torch.tensor(criterion.ignore_index).type_as(target)
    #     )
    #     loss = criterion(active_logits, active_labels)

    #     return loss

    def forward(self, ids, mask, seg_ids, labels):
        o1 = self.biobert_layer(
            ids,
            attention_mask=mask,
            token_type_ids=seg_ids
        )
        # print(o1)
        cont_reps = o1.last_hidden_state
        cls_rep = cont_reps[:, 0]
        out = self.dropout(cls_rep)
        out = self.output(out)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out.squeeze(-1), labels.float())

        return out, loss

if __name__ == "__main__":
    pass
