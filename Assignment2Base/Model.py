import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##### General functions for Models #####
def train_model(model, epochs, batch_size, iterator_train, iterator_validation, optimizer, loss, num_train_samples):
    model.to(device)
    num_iters = num_train_samples // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        epoch_loss_val = 0
        epoch_acc_val = 0
        for i, (inputs, labels) in enumerate(iterator_train):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            predictions = model(inputs).squeeze(1)
            loss_iter = loss(predictions, labels)
            acc_iter = binary_accuracy(predictions, labels)
            loss_iter.backward()
            optimizer.step()
            epoch_loss += loss_iter.item()
            epoch_acc += acc_iter.item()
            if (i + 1) % (num_iters // 10) == 0:
                print(f"Epoch: {epoch + 1}/{epochs} -- Step: {i + 1}/{num_iters} "
                      f"-- Acc: {acc_iter.item():.2F} -- Loss: {loss_iter.item():.2F}")

        for i, (inputs, labels) in enumerate(iterator_validation):
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            with torch.no_grad():
                predictions = model(inputs).squeeze(1)
                loss_iter = loss(predictions, labels)
                acc_iter = binary_accuracy(predictions, labels)
                epoch_loss_val += loss_iter.item()
                epoch_acc_val += acc_iter.item()

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
        self.embedding = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=True, padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.fc = nn.Linear(100, 1)
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # claim = [max_tok, emb_dim]
        # evid = [max_tok, emb_dim]
        # print("input shape:", x.shape)

        # print(x[:, 0].long())
        emb_claim = self.embedding(x[:, 0].long())
        emb_evid = self.embedding(x[:, 1].long())
        # print("emb_merda:", emb_evid.shape)  # [32, 90, 50]

        output_claim, hidden_claim = self.rnn(emb_claim)
        output_evid, hidden_evid = self.rnn(emb_evid)
        # print("hidden merda:", hidden_claim.shape)  # [1, 1, 50]

        concat = torch.cat((hidden_claim[0], hidden_evid[0]), 1)
        # print("concat merda", concat.shape)  # [1, 1, 100]
        # print(concat)
        dense = self.fc(concat)
        return dense
