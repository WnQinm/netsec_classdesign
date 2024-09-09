import torch
import torch.nn as nn
from torch.nn import LSTM, CrossEntropyLoss
from transformers import BertTokenizer, BertModel


class Model(nn.Module):

    def __init__(self, model_path=None, num_labels=5, freeze_backbone_params=False):
        super().__init__()
        self.freeze_backbone_params = freeze_backbone_params
        self.load_model(model_path)
        hidden_size = self.model.config.hidden_size
        self.lstm = LSTM(hidden_size, hidden_size, batch_first=True)
        self.projection_head = nn.Linear(self.model.config.hidden_size, num_labels)
        self.loss_func = CrossEntropyLoss()

    def load_model(self, model_path) -> None:
        if model_path:
            self.model = BertModel.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if self.freeze_backbone_params:
            for k,v in self.model.named_parameters():
                v.requires_grad = False

    def encode(self, x):
        x = self.model(**x).last_hidden_state[:, 0]
        x, _ = self.lstm(x)
        x = x[-1, :]
        x = self.projection_head(x)
        return x

    def forward(self, inputs):
        label, data = inputs
        data = torch.stack(list(map(self.encode, data)))
        return self.loss_func(data, label)

    def save(self, output_dir: str):
        _trans_state_dict = lambda state_dict: type(state_dict)({k: v.clone().cpu() for k,v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(self.model.state_dict()))
