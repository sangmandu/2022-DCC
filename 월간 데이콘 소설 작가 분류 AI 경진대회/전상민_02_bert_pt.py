from transformers import BertTokenizerFast, BertForSequenceClassification, BertForMaskedLM, AutoConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from tqdm import tqdm
from huggingface_hub import notebook_login, HfApi
from easydict import EasyDict

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss

from torch.utils.data import Dataset, DataLoader
import torch

import os
import click
import multiprocessing
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
        
        
@click.command()
@click.option('--use_ptr_tok', default=False, help='whether using pre-trained tokenizer or pre-train tokenizer directly')
@click.option('--ptr_model', default=True, help='whether pretraining model directly or just loading pre-trained model')
@click.option('--csv_name', required=True, metavar='STR', help='name of csv file that test dataset is predicted on')

def main(**kwargs):
    opts = EasyDict(kwargs)
    
    try:
        notebook_login()
    except:
        class HuggingFaceLoginError(Exception):
            def __str__(self):
                return "enter 'huggingface-cil login' and tokens"
        raise HuggingFaceLoginError
            
    user_id = HfApi().whoami()["name"]
    print(f"user id '{user_id}' will be used during the example")

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_x.csv')
    submit = pd.read_csv('sample_submission.csv')

    tokenizer_id = "dacon-novelauthor" 
    if not opts.use_ptr_tok:
        try:
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_id)
        except:
            def batch_iterator(batch_size=10000):
                for i in tqdm(range(0, len(train_df), batch_size)):
                    yield train_df[i : i + batch_size]["text"]

            tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-small")
            bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=30522)
            bert_tokenizer.save_pretrained("tokenizer")
            bert_tokenizer.push_to_hub(tokenizer_id)
    else:
        tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-small")
    
    print(
        tokenizer.tokenize("what the fucking [MASK]. what's up?")
    )
    
    config = AutoConfig.from_pretrained('prajjwal1/bert-small')
    config.num_labels=5
    
    output_dir = 'model_output'
    checkpoint = 'checkpoint-8000'
    
    if opts.ptr_model:
        for file in os.listdir(output_dir):
            if checkpoint == file:
                break
        else:
            tokenized_text = tokenizer(
                list(train_df.text.values),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_token_type_ids = True
            )
            train_dataset = CustomDataset(tokenized_text)

            model = BertForMaskedLM(config=config)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, 
                mlm=True,
                mlm_probability= 0.15
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=10,
                per_device_train_batch_size=16,
                save_steps=1000, # step 수마다 모델을 저장
                save_total_limit=2, # 마지막 두 모델 빼고 과거 모델은 삭제
                logging_steps=100,
                gradient_accumulation_steps=4,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator, # 밥을 어떻게 떠먹여줄지
                train_dataset=train_dataset # 밥이 뭔지
            )

            trainer.train()
            trainer.save_model('.')
        
        model_dir = os.path.join(output_dir, checkpoint)
        model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    
    else:
        model = BertForSequenceClassification(config=config)

    
    x, y = train_df.text, train_df.author    
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
    
    train_dataset = CustomDataset(tokenized_train_x, train_y)
    valid_dataset = CustomDataset(tokenized_valid_x, valid_y)
    test_dataset = CustomDataset(tokenized_test_x)
    
    training_args = TrainingArguments(
        output_dir='./fine-tuning',  # output directory
        save_total_limit=2,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=10,  # total number of training epochs
        learning_rate=3e-5,  # learning_rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=128,  # batch size for evaluation
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        evaluation_strategy='steps',  # evaluation strategy to adopt during training
        eval_steps=500,  # evaluation step.
        metric_for_best_model='logloss',
        greater_is_better='False',
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
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

    trainer.train()


    print(
        trainer.evaluate()
    )


    def softmax(a):
        exp_a = np.exp(a)
        return exp_a / np.expand_dims(np.sum(exp_a, axis=1), 1)

    preds = trainer.predict(test_dataset)
    preds = softmax(preds.predictions)

    submit.iloc[:, 1:6] = preds
    submit.to_csv(opts.csv_name, index=False)

    
if __name__ == '__main__':
    main()
    
    