import torch
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder

from Dataset.dataset import SRC, TRG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 5e-5
num_epochs = 20
batch_size = 16

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 384
ENC_LAYERS = 8
DEC_LAYERS = 8
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 4 * HID_DIM
DEC_PF_DIM = 4 * HID_DIM
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25


enc_par = { 'hid_dim': 384,
            'n_layers': 6,
            'n_heads': 8,
            'pf_dim': 2 * 384,
            'dropout': 0.25}

dec_par = { 'hid_dim': 384,
            'n_layers': 6,
            'n_heads': 8,
            'pf_dim': 2 * 384,
            'dropout': 0.25}

encoder = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

decoder = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)


