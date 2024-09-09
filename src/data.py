from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DataCollatorWithPadding


LABELS = ['chat', 'email', 'file_down', 'file_up', 'stream']


class CustomDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._dataset = load_dataset("json", data_files=data_path, split='train')
        self._dataset = self._dataset.train_test_split(test_size=0.001)

    @property
    def train(self):
        self.dataset = self._dataset['train']
        return self

    @property
    def test(self):
        self.dataset = self._dataset['test']
        return self

    def __getitem__(self, index):
        data = self.dataset[index]
        return data['label'], data['data']

    def __len__(self):
        return len(self.dataset)


class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, features):
        label, data = zip(*features)
        label = torch.tensor(list(map(lambda x: LABELS.index(x), label)))
        data = list(map(self.tokenize, data))
        return label, data