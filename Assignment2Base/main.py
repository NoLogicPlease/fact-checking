import torch
from torch import optim, nn

from torch.utils.data import DataLoader
from Tokenizer import Tokenizer
from FactDataset import FactDataset
import gensim
import gensim.downloader as gloader
import pandas as pd
import pickle
from Model import Model
from dataset_tools import generator

# GLOBALS
EMBEDDING_DIMENSION = 50
BATCH_SIZE = 32
NUM_CLASSES = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# LOAD CLEANED DATA
train = FactDataset(EMBEDDING_DIMENSION, glove_dict, glove_matrix, 'train')
val = FactDataset(EMBEDDING_DIMENSION, glove_dict, glove_matrix, 'val')
test = FactDataset(EMBEDDING_DIMENSION, glove_dict, glove_matrix, 'test')

dataloader_train = DataLoader(dataset=train, batch_size=2, shuffle=True, num_workers=4)
dataloader_val = DataLoader(dataset=val, batch_size=2, shuffle=True, num_workers=4)
dataloader_test = DataLoader(dataset=test, batch_size=2, shuffle=True, num_workers=4)

dataiter = iter(dataloader_train)
data = dataiter.next()
features, labels = data
print(features)
print(labels)

quit()

max_sen_len = 0

model = Model(max_sen_len, EMBEDDING_DIMENSION, NUM_CLASSES)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
