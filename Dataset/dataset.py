import re
import pandas as pd
import torch
from spacy.lang.en import English
from spacy.lang.ar import Arabic
from spacy.tokenizer import Tokenizer
from torchtext.legacy.data import BucketIterator, Field, Dataset, Example

enNLP = English()
arNLP = Arabic()
enTokenizer = Tokenizer(enNLP.vocab)
arTokenizer = Tokenizer(arNLP.vocab)


def myTokenizerEN(x):
    return [word.text for word in
            enTokenizer(re.sub(r"\s+\s+", " ", re.sub(r"[\.\'\`\"\r+\n+]", " ", x.lower())).strip())]


def myTokenizerAR(x):
    return [word.text for word in
            arTokenizer(re.sub(r"\s+\s+", " ", re.sub(r"[\.\'\`\"\r+\n+]", " ", x.lower())).strip())]


mydata = pd.read_csv("Dataset/ara_eng.txt", delimiter="\t", names=["eng", "ar"])
SRC = Field(tokenize=myTokenizerEN, batch_first=True, init_token="<sos>", eos_token="<eos>")
TRG = Field(tokenize=myTokenizerAR, batch_first=True, tokenizer_language="ar", init_token="ببدأ", eos_token="نهها")


class TranslateData(Dataset):
    def __init__(self, dataset, src_field, target_field, is_test=False, **kwargs):
        fields = [('eng', src_field), ('ar', target_field)]
        examples = []
        for i, row in dataset.iterrows():
            eng = row.eng
            ar = row.ar
            examples.append(Example.fromlist([eng, ar], fields))
        super().__init__(examples, fields, **kwargs)


data = TranslateData(mydata, SRC, TRG)
train_data, valid_data = data.split(split_ratio=0.8)
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print('finished loading dataset!')
if __name__ == '__main__':
    # print(SRC.vocab.stoi[' '])
    # print(TRG.vocab.stoi[' '])
    BATCH_SIZE = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, validation_iterator = BucketIterator.splits(
        (train_data, valid_data), shuffle=True, batch_size=BATCH_SIZE, device=device, sort=False)
    for batch in validation_iterator:
        x = batch.eng
        y = batch.ar
        import ipdb;ipdb.set_trace()

