import os.path

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchsummary import summary

import pickle
import gensim
import gensim.downloader as gloader
import pandas as pd

from Model import Model_RNN1, train_model
from Tokenizer import Tokenizer
from FactDataset import FactDataset

# GLOBALS
EMBEDDING_DIMENSION = 50
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 10


def load_dataset():
    # LOAD GLOVE
    try:
        with open(f"glove-{EMBEDDING_DIMENSION}.pkl", 'rb') as f:
            emb_model = pickle.load(f)
    except Exception:
        emb_model = gloader.load(f"glove-wiki-gigaword-{EMBEDDING_DIMENSION}")
        with open(f"glove-{EMBEDDING_DIMENSION}.pkl", 'wb') as f:
            pickle.dump(emb_model, f)

    glove_dict = emb_model.key_to_index
    glove_matrix = emb_model.vectors

    # LOAD CLEANED DATA IN TORCH DATASET OBJECTS
    splits = {}
    for split in ['train', 'val', 'test']:
        try:
            with open(f"{os.path.join('dataset_torched', split)}.pkl", 'rb') as f:
                splits[split] = pickle.load(f)
        except Exception:
            splits[split] = FactDataset(EMBEDDING_DIMENSION, glove_dict, glove_matrix, split)
            with open(f"{os.path.join('dataset_torched', split)}.pkl", 'wb') as f:
                pickle.dump(splits[split], f)

    return splits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

splits = load_dataset()
train, val, test = splits['train'], splits['val'], splits['test']
dataloader_train = DataLoader(dataset=train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
dataloader_val = DataLoader(dataset=val, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
dataloader_test = DataLoader(dataset=test, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

dataiter_train = iter(dataloader_train)
dataiter_val = iter(dataloader_val)
dataiter_test = iter(dataloader_test)

# print(torch.tensor(train.emb_matrix, device=device))
print(max(train.val_to_key.values()))
print(train.emb_matrix.shape)

model_params = {
    'sentence_len': train.max_seq_len,
    'embedding_dim': EMBEDDING_DIMENSION,
    'output_dim': NUM_CLASSES,
    'pre_trained_emb': torch.tensor(train.emb_matrix).to(device)
}
model = Model_RNN1(**model_params)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss = nn.BCEWithLogitsLoss()

model = model.to(device)
loss = loss.to(device)

# summary(model, (2, 90), batch_size=32)
# quit()

training_info = {
    'model': model,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'iterator_train': dataiter_train,
    'iterator_validation': dataiter_val,
    'optimizer': optimizer,
    'loss': loss,
    'num_train_samples': train.n_samples
}

train_model(**training_info)
