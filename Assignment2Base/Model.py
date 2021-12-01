import torch.nn as nn
import torch
import torch.optim as optim


##### General functions for Models #####
def train_model(model, epochs, batch_size, iterator_train, iterator_validation, optimizer, loss, num_train_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_iters = num_train_samples // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        epoch_loss_val = 0
        epoch_acc_val = 0
        for i, (inputs, labels) in enumerate(iterator_train):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = loss(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if (i + 1) % (num_iters // 5) == 0:
                print(f"Epoch: {epoch + 1}/{epochs} -- Step: {i + 1}/{num_iters} "
                      f"-- Acc: {acc:.2F} -- Loss: {loss:.2F}", flush=True)

        for i, (inputs, labels) in enumerate(iterator_validation):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                predictions = model(inputs)
                loss = loss(predictions, labels)
                acc = binary_accuracy(predictions, labels)
                epoch_loss_val += loss.item()
                epoch_acc_val += acc.item()

        print(f"Epoch: {epoch + 1}/{epochs} -- Step: {num_iters}/{num_iters} "
              f"-- Acc: {epoch_acc / len(iterator_train):.2F} "
              f"-- Loss: {epoch_loss / len(iterator_train):.2F} "
              f"-- Acc_Val: {epoch_acc_val / len(iterator_validation):.2F} "
              f"-- Loss_Val: {epoch_loss_val / len(iterator_validation)}:.2F")

    return epoch_acc / len(iterator_train), epoch_loss / len(iterator_train)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


#########################################

class Model_RNN1(nn.Module):
    def __init__(self, sentence_len, embedding_dim, output_dim, pre_trained_emb):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=True, padding_idx=-1)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=sentence_len)
        self.fc = nn.Linear(embedding_dim * 2, output_dim)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # claim = [max_tok, emb_dim]
        # evid = [max_tok, emb_dim]
        emb_claim = self.embedding(x[0])
        emb_evid = self.embedding(x[1])
        output_claim, hidden_claim = self.rnn(emb_claim)
        output_evid, hidden_evid = self.rnn(emb_evid)
        concat = torch.cat((hidden_claim, hidden_evid), 1)
        #TODO: check from here the squeezeee!!!!
        dense = self.fc(concat.squeeze(0))
        softmax = self.softmax(dense)
        return softmax
