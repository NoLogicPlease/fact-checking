import torch.nn as nn
import torch
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, sentence_len, embedding_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=sentence_len, hidden_size=embedding_dim)
        self.fc = nn.Linear(embedding_dim*2, output_dim)

    def forward(self, claim, evid):
        # claim = [max_tok, emb_dim]
        # evid = [max_tok, emb_dim]

        output_claim, hidden_claim = self.rnn(claim)
        output_evid, hidden_evid = self.rnn(evid)
        concat = torch.cat((hidden_claim, hidden_evid), 1)

        return self.fc(concat.squeeze(0))


