import copy
import numpy as np
from tqdm import tqdm


class Tokenizer(object):
    def __init__(self, dataset_sentences, embedding_dim, glove_dict, glove_matrix):
        self.embedding_matrix = None
        self.value_to_key = {}
        self.value_to_key_new = {}
        self.key_to_value = {}
        self.num_unique_words = 0
        self.dataset_sentences = dataset_sentences
        self.embedding_dim = embedding_dim
        self.glove_dict = glove_dict
        self.glove_matrix = glove_matrix
        self.unique_words = set()

    def get_val_to_key(self):
        return copy.deepcopy(self.value_to_key)

    def tokenize(self):
        self.value_to_key_new = {}
        unique_words = set()
        for sen in self.dataset_sentences:
            for w in sen.split():
                unique_words.add(w)  # get se of unique words
        new_unique = unique_words - self.unique_words
        for i, word in enumerate(new_unique):
            if self.embedding_matrix is not None:
                self.key_to_value[i + len(self.embedding_matrix)] = word  # build two dictionaries for key value correspondence
                self.value_to_key[word] = i + len(self.embedding_matrix)
                self.value_to_key_new[word] = i
            else:
                self.key_to_value[i] = word  # build two dictionaries for key value correspondence
                self.value_to_key[word] = i

        self.num_unique_words = len(new_unique)
        self.unique_words = self.unique_words | new_unique  # union of unique words and new unique words


    def __build_embedding_matrix_glove(self):
        oov_words = []
        tmp_embedding_matrix = np.zeros((self.num_unique_words, self.embedding_dim), dtype=np.float32)
        len_old_emb_matrix = len(self.embedding_matrix) if self.embedding_matrix is not None else 0
        for word, idx in tqdm(self.value_to_key_new.items()):
            try:
                embedding_vector = self.glove_matrix[self.glove_dict[word]]
                tmp_embedding_matrix[idx] = embedding_vector
            except (KeyError, TypeError):
                oov_words.append((word, idx + len_old_emb_matrix))
        if self.embedding_matrix is not None:
            self.embedding_matrix = np.vstack((self.embedding_matrix, tmp_embedding_matrix))
        else:
            self.embedding_matrix = tmp_embedding_matrix
        return oov_words

    def build_embedding_matrix(self):
        oov_words = self.__build_embedding_matrix_glove()
        for word, idx in oov_words:
            neighbour_words = []
            for sen in self.dataset_sentences:  # look for word in sentence
                for i, wanted_word in enumerate(sen):
                    if wanted_word == word:
                        neighbour_words.append(sen[i - 1])  # append previous word in list of neighbours
                        neighbour_words.append(sen[i + 1])  # append next word in list of neighbours
            avg_matrix = np.zeros((len(neighbour_words), self.embedding_dim))  # initialize matrix of avgs

            length_in_vocab = 0  # to check if neighbours are OOV
            for i, el in enumerate(neighbour_words):
                try:
                    avg_matrix[i] = self.embedding_matrix[self.value_to_key[el]]  # check not OOV
                    length_in_vocab += 1  # we don't want to use the zero columns of avg_matrix
                except (KeyError, TypeError):  # the model doesn't exist
                    pass
            if length_in_vocab == 0:
                embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=self.embedding_dim)
            else:
                embedding_vector = np.mean(avg_matrix[:length_in_vocab], axis=0)
            self.embedding_matrix[idx] = embedding_vector
        return copy.deepcopy(self.embedding_matrix)
