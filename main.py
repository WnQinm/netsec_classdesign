from src.modeling import Model
from src.trainer import CustomTrainer
from src.data import CustomCollator, CustomDataset
from src.utils import compute_metrics

import os
import json

from transformers import set_seed, TrainingArguments, HfArgumentParser


ARG_PATH = "./arguments.json"
set_seed(42)


if __name__ == "__main__":
    with open(ARG_PATH, 'r') as f:
        training_args = TrainingArguments(**json.load(f))
    model = Model(freeze_backbone_params=True)
    model.train()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=CustomDataset("res.json").train,
        eval_dataset=CustomDataset("res.json").test,
        data_collator=CustomCollator(model.tokenizer),
        tokenizer=model.tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
