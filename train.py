import math
import os
import time

import torch
from torch import nn
from torchtext.legacy.data import BucketIterator
from tqdm import tqdm

from Dataset.dataset import SRC, TRG, valid_data, train_data
from configs import encoder, decoder, learning_rate, num_epochs, batch_size
from Transformer.Translator import Seq2Seq


def train(model, iterator, optimizer, Loss):
    model.train()
    epoch_loss = 0
    for batch in tqdm(iterator):
        src = batch.eng
        trg = batch.ar
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = Loss(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, Loss):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.eng
            trg = batch.ar
            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = Loss(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()
    return dir_num


if __name__ == '__main__':
    if not os.path.isdir('experiments'):
        os.mkdir('experiments')
    dir_num = -1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Loss = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_valid_loss = float('inf')

    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), shuffle=True, batch_size=batch_size, device=device, sort=False)

    if dir_num == -1:
        ds = os.listdir('experiments')
        if len(ds) == 0:
            dir_num = 1
        else:
            ds = list(map(int, ds))
            dir_num = max(ds) + 1
        os.mkdir('experiments/'+str(dir_num))
    save_vocab(SRC.vocab, f'experiments/{dir_num}/src.pkl')
    save_vocab(TRG.vocab, f'experiments/{dir_num}/trg.pkl')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, Loss)
        valid_loss = evaluate(model, valid_iterator, Loss)
        scheduler.step()

        end_time = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'experiments/{dir_num}/model.pth')

        print(f'Epoch: {epoch + 1:02} | Time: {end_time - start_time}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('________________________________________________________________________')


