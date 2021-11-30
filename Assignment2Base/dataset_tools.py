import numpy as np
import random


def generator(data, batch_size, tokenizer, embedding_dim):
    """

    :param data: Already embedded with Glove [ [max_tokens], [max_tokens]... ]
    :param batch_size:
    :return:
    """

    data.sample(frac=1)  # rows random shuffle
    i = 0
    X = np.zeros((batch_size, data.shape[1], embedding_dim))
    y = np.zeros(batch_size)
    dataset_size = len(data)

    while True:
        for j in range(i, i + batch_size):
            claim = data['Claim'][i]
            evid = data['Evidence'][i]
            claim_emb = [tokenizer.embedding_matrix[tokenizer.value_to_key[word]] for word in claim]
            evid_emb = [tokenizer.embedding_matrix[tokenizer.value_to_key[word]] for word in evid]
            X[j - i] = [claim_emb, evid_emb]
            y[j - i] = data['Label']
        i += batch_size
        if (i + batch_size >= dataset_size):
            i = 0
            data.sample(frac=1)
        yield X, y
