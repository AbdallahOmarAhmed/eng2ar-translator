import argparse
import os.path
import pickle

from Transformer.Translator import Seq2Seq
import torch

from configs import enc_par, dec_par
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder


def translate_sentence(sentence, SRC, TRG, model, device, max_len=99):
    model.eval()

    sentence = sentence.split()
    tokens = [token.lower() for token in sentence]

    tokens = [SRC.itos[2]] + tokens + [SRC.itos[3]]

    src_indexes = [SRC.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [2]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        if pred_token == 3:
            break
        if pred_token != 0:
            trg_indexes.append(pred_token)

    trg_tokens = [TRG.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_num", type=int)
    args = parser.parse_args()
    path = 'experiments/'+str(args.exp_num)

    SRC = pickle.load(open(os.path.join(path, 'src.pkl'), 'rb'))
    TRG = pickle.load(open(os.path.join(path, 'trg.pkl'), 'rb'))

    enc_par.update({'device': device, 'input_dim': len(SRC)})
    dec_par.update({'device': device, 'output_dim': len(TRG)})

    encoder = Encoder(**enc_par)
    decoder = Decoder(**dec_par)

    model = Seq2Seq(encoder, decoder, 1, 1, device).to(device)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    model.to(device)

    while True:
        sen = input('enter a sentence: ')
        out, _ = translate_sentence(sen, SRC, TRG, model, device)
        print(out)