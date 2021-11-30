import torch
from torch import optim, nn

from Tokenizer import Tokenizer
import gensim
import gensim.downloader as gloader
import pandas as pd
import pickle
from Model import Model
from dataset_tools import generator

EMBEDDING_DIMENSION = 50
BATCH_SIZE = 32
NUM_CLASSES = 2

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
train_pairs = pd.read_csv("dataset_cleaned/train_pairs.csv")
val_pairs = pd.read_csv("dataset_cleaned/val_pairs.csv")
test_pairs = pd.read_csv("dataset_cleaned/test_pairs.csv")

train_text = train_pairs['Evidence'] + train_pairs['Claim']
val_text = val_pairs['Evidence'] + val_pairs['Claim']
test_text = test_pairs['Evidence'] + test_pairs['Claim']

tokenizer = Tokenizer(train_text, EMBEDDING_DIMENSION, glove_dict, glove_matrix)
tokenizer.tokenize()
v2_val_to_key = tokenizer.get_val_to_key()
v2_matrix = tokenizer.build_embedding_matrix()

tokenizer.dataset_sentences = val_text
tokenizer.tokenize()
v3_matrix = tokenizer.build_embedding_matrix()
v3_val_to_key = tokenizer.get_val_to_key()

tokenizer.dataset_sentences = test_text
tokenizer.tokenize()
v4_matrix = tokenizer.build_embedding_matrix()
v4_val_to_key = tokenizer.get_val_to_key()

max_sen_len = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(max_sen_len, EMBEDDING_DIMENSION, NUM_CLASSES)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
model = model.to(device)
criterion = criterion.to(device)


train_generator = next(generator(train_pairs, BATCH_SIZE, tokenizer, EMBEDDING_DIMENSION))



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
