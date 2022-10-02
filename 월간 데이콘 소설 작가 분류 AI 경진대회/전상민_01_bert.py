from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

from torch.utils.data import Dataset, DataLoader
import torch

import re
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.x.items()}
        if self.y is not None:
            item['labels'] = torch.tensor(self.y.iloc[idx])
        return item

    def __len__(self):
        return len(self.x['input_ids'])
    
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')
submit = pd.read_csv('sample_submission.csv')

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
config = AutoConfig.from_pretrained('bert-base-cased')
config.num_labels=5

model = BertForSequenceClassification(config=config)
x, y = df.text, df.author

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=1004)

tokenized_train_x = tokenizer(
    list(train_x.values),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    return_token_type_ids = True
)

tokenized_valid_x = tokenizer(
    list(valid_x.values),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    return_token_type_ids = True
)

tokenized_test_x = tokenizer(
    list(test_df.text.values),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
    return_token_type_ids = True
)

# encoder = OneHotEncoder(sparse=False)

# oh_train_y = encoder.fit_transform(np.reshape(train_y.values, (-1, 1)))
# oh_valid_y = encoder.fit_transform(np.reshape(valid_y.values, (-1, 1)))

train_dataset = CustomDataset(tokenized_train_x, train_y)
valid_dataset = CustomDataset(tokenized_valid_x, valid_y)
test_dataset = CustomDataset(tokenized_test_x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

training_args = TrainingArguments(
        output_dir='./',  # output directory
        save_total_limit=2,  # number of total save model.
        save_steps=200,  # model saving step.
        num_train_epochs=10,  # total number of training epochs
        learning_rate=3e-5,  # learning_rate
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        eval_steps=200,  # evaluation step.
        metric_for_best_model='logloss',
        greater_is_better='False',
        load_best_model_at_end=True,
        gradient_accumulation_steps=16,
    )

def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions
    return {
        'logloss': log_loss(labels, preds, labels=list(range(config.num_labels)))
    }

                    
trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # early stopping
    )

trainer.train("checkpoint-1500")


print(
    trainer.evaluate()
)


def softmax(a):
    exp_a = np.exp(a)
    return exp_a / np.expand_dims(np.sum(exp_a, axis=1), 1)

preds = trainer.predict(test_dataset)
preds = softmax(preds.predictions)

submit.iloc[:, 1:6] = preds
submit.to_csv('result_base.csv', index=False)
