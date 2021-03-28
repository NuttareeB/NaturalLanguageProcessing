import torch
import torch.nn as nn

from torchtext import data
from torchtext import datasets

import numpy as np
import spacy
from matplotlib import pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    # dataset
    TEXT = data.Field(lower = True) 
    UD_TAGS = data.Field(unk_token = None)
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data , valid_data , test_data = datasets.UDPOS.splits(fields)
    TEXT.build_vocab(train_data, vectors = "glove.6B.100d")
    UD_TAGS.build_vocab(train_data)
    
    # variable
    batch_size = 128
    embedding_dim = 100
    input_dim = len(TEXT.vocab)
    output_dim = len(UD_TAGS.vocab)
    hidden_dim = 128
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    tag_pad_idx = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                    (train_data, valid_data, test_data), 
                                                    batch_size = batch_size,
                                                    device = device)

    # model
    model = BiDirectLSTM(input_dim, output_dim, hidden_dim, embedding_dim, pad_idx)
    model.to(device)

    # embedding
    glove_pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(glove_pretrained_embeddings)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

    # optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.000001)
    criterion = nn.CrossEntropyLoss(ignore_index = tag_pad_idx)
    criterion = criterion.to(device)

    train_losses = []
    val_losses = []

    lowest_val_loss = float('inf')

    # train model and evaluate the model with train's parameter
    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion, tag_pad_idx)
        val_loss, val_acc = evaluate_model(model, valid_iterator, criterion, tag_pad_idx)
                                           
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pt')

        print("epoch %d \ttrain loss %.3f, train acc %.3f%% \tval loss %.3f, val acc %.3f%%" % (epoch+1, train_loss, train_acc*100, val_loss, val_acc*100))

    # evaluate test set
    model.load_state_dict(torch.load('model.pt'))
    test_loss, test_acc = evaluate_model(model, test_iterator, criterion, tag_pad_idx)
    print("test loss %.3f, test acc %.3f%%" % (test_loss, test_acc*100))

    sentences = [["the", "old", "man", "the", "boat", "."], 
    ["The", "complex", "houses", "married", "and", "single", "soldiers", "and", "their", "families", "."],
    ["The", "man", "who", "hunts", "ducks", "out", "on", "weekends", "."]]
    for sentence in sentences:
        tokens, predicted_tags = tag_sentence(model, sentence, TEXT, UD_TAGS)
        print("-----------------------------")
        print("TAG\t\t TOKEN\n")
        for token, tag in zip(tokens, predicted_tags):
            print(tag, "\t\t", token)
    
    plot_loss(train_losses, val_losses)

def plot_loss(train_losses, val_losses):
    plt.title("Training and validation loss over the course of training.")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["training", "validation"], loc ="upper right")
    # plt.xticks(np.arange(len(xticks)), xticks)
    plt.grid(linestyle='-')
    plt.show()

class BiDirectLSTM(torch.nn.Module) :

    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim, pad_idx) :
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)

        self.lstm = nn.LSTM(input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True)

        self.linear = nn.Linear(hidden_dim * 2, output_dim)   

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (ht, _) = self.lstm(x)
        out = self.linear(lstm_out)
        return out

def train_model(model, iterator, optimizer, criterion, tag_pad_idx):
    
    sum_loss = 0
    sum_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.udtags
        
        optimizer.zero_grad()
        
        preds = model(text)
        preds = preds.view(-1, preds.shape[-1])
        tags = tags.view(-1)
        loss = criterion(preds, tags)
        acc = calculate_accuracy(preds, tags, tag_pad_idx)
        loss.backward()
        
        optimizer.step()
        
        sum_loss += loss.item()
        sum_acc += acc.item()
        
    return sum_loss/len(iterator), sum_acc/len(iterator)

def evaluate_model(model, iterator, criterion, tag_pad_idx):
    
    sum_loss = 0
    sum_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:

            text = batch.text
            tags = batch.udtags
            
            preds = model(text)
            preds = preds.view(-1, preds.shape[-1])
            tags = tags.view(-1)
            loss = criterion(preds, tags)
            acc = calculate_accuracy(preds, tags, tag_pad_idx)

            sum_loss += loss.item()
            sum_acc += acc.item()
        
    return sum_loss/len(iterator), sum_acc/len(iterator)

def calculate_accuracy(preds, y, tag_pad_idx):
    non_pads = torch.nonzero(y != tag_pad_idx)

    max_preds = preds.argmax(dim = 1, keepdim = True)
    predictions = max_preds[non_pads].squeeze(1)
    ground_truths = y[non_pads]
    correct = predictions.eq(ground_truths)
    sum_corr = correct.sum()
    total = torch.FloatTensor([y[non_pads].shape[0]]).to(device)

    return sum_corr/total

def tag_sentence(model, tokens, text, ud_tag):
    
    model.eval()
        
    numericalized_tokens = [text.vocab.stoi[t] for t in tokens]
    
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1).to(device)
         
    preds = model(token_tensor)
    
    top_preds = preds.argmax(-1)
    
    predicted_tags = [ud_tag.vocab.itos[t.item()] for t in top_preds]
    
    return tokens, predicted_tags


if __name__== "__main__":
    main()