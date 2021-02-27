import torch
from trns.vocab import Vocab
from trns.nmt_model import NMT
from trns.preproc_En import pre_en
from trns.preproc_kor import preproc_ko2en
from trns.trns_koren import Trns

def trns_model():
    pre_ko = preproc_ko2en()
    vocab_trns = Vocab.load('trns/vocab.json')
    model = NMT(vocab=vocab_trns, embed_size=300, hidden_size=300, dropout_rate=0.0)
    model.load_state_dict(torch.load('trns/model_bi_0811', map_location=lambda storage, loc: storage))
    model.eval()
    trns = Trns(model)
    
    return pre_en, pre_ko, trns
