"""
model_train - 训练模型

Author: 骆昊
Version: 0.0.1
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.data_pre import get_corpus


class TMFDataset(Dataset):

    def __init__(self, corpus, tokenizer, max_len=32):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        doc, label = self.corpus[index]
        inputs = self.tokenizer(
            doc,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model():
    """模型训练"""
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_model)

    train_corpus = get_corpus(Config.train_raw_file)
    random.seed(3)
    samples = random.sample(train_corpus, k=900)

    train_dataset = TMFDataset(samples, tokenizer, max_len=32)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(Config.pretrained_model, num_labels=10)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    EPOCHS = 16
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_func(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch[{epoch}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}')

    model.save_pretrained(Config.model_output_dir)
    tokenizer.save_pretrained(Config.model_output_dir)
