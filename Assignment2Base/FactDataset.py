import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from Tokenizer import Tokenizer
from tqdm import tqdm


class FactDataset(Dataset):
    def __init__(self, emb_dim, glove_dict, glove_matrix, split='train'):
        pairs_claim_evid = pd.read_csv(f"dataset_cleaned/{split}_pairs.csv")
        # tokenization & embeddings
        text = pairs_claim_evid['Claim'] + ' ' + pairs_claim_evid['Evidence']
        text = text.astype(str).to_list()

        self.tokenizer = Tokenizer(text, emb_dim, glove_dict, glove_matrix)
        self.tokenizer.tokenize()
        self.val_to_key = self.tokenizer.get_val_to_key()
        self.emb_matrix = self.tokenizer.build_embedding_matrix()
        lengths = pairs_claim_evid['Evidence'].str.split().str.len().to_numpy()

        # print(len([i for i, el in enumerate(lengths) if el > 100]))

        self.max_claim_len = int(pairs_claim_evid['Claim'].str.split().str.len().max())
        self.max_evid_len = int(pairs_claim_evid['Evidence'].str.split().str.len().max())
        self.max_seq_len = int(max(self.max_evid_len, self.max_claim_len))

        if not self._check_tokenizer():
            raise ValueError

        # create x as list of sentences
        # sentences are splitted in claim evidence
        self.x = []
        print(f"Creating FactDataset: \n")
        for i, (claim_sen, evid_sen) in tqdm(enumerate(zip(pairs_claim_evid['Claim'], pairs_claim_evid['Evidence']))):
            sentence = [[], []]
            for word in claim_sen.split():
                sentence[0].append(self.val_to_key[word])
            for word in evid_sen.split():
                sentence[1].append(self.val_to_key[word])

            sentence[0] = sentence[0] + [-1] * (self.max_seq_len - len(claim_sen.split()))
            sentence[1] = sentence[1] + [-1] * (self.max_seq_len - len(evid_sen.split()))
            self.x.append(sentence)

        self.x = torch.tensor(self.x)
        self.y = torch.tensor(pairs_claim_evid['Label'])
        self.n_samples = pairs_claim_evid.shape[0]
        print(f"Max sequence length: {self.max_seq_len}")

    def __getitem__(self, index):
        # xi = [[claim][evid]], yi = [label]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def _check_tokenizer(self):
        word = 'the'
        for i, el in enumerate(self.tokenizer.glove_matrix[self.tokenizer.glove_dict[word]]):
            if el != self.emb_matrix[self.val_to_key[word]][i]:
                print("Check Tokenizer, possible bugs")
                return False
        return True
