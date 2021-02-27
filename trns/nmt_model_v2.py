#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from trns.utils import get_sents_lenth4_new, get_sents_lenth_new
from itertools import chain

from trns.model_embeddings import ModelEmbeddings
Ehypothesis = namedtuple('Ehypothesis', ['value', 'score'])
Khypothesis = namedtuple('Khypothesis', ['value', 'xo', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.token = ['(', ')', ',', "'", '"','_','<s>','</s>']
        self.sbol = ['_','^','`']
        self.xo_weight = 0.4
        #self.sbol_padded = self.vocab.vocs.to_input_tensor([['_'],['^'],['`']], device=self.device)

        # default values
        self.ek_encoder = None 
        self.ek_decoder = None
        self.ek_h_projection = None
        self.ek_c_projection = None
        self.ek_att_projection = None
        self.ek_combined_output_projection = None
        self.ek_target_vocab_projection = None
        
        self.ke_encoder = None 
        self.ke_decoder = None
        self.ke_h_projection = None
        self.ke_c_projection = None
        self.ke_att_projection = None
        self.ke_combined_output_projection = None
        self.ke_target_vocab_projection = None    
        
        self.dropout = None
        """
        self.sub_encoder= None
        self.en_gate = None
        self.sub_en_projection = None  
        
        self.sub_decoder= None
        self.de_gate = None
        self.sub_de_projection = None
        """
        self.sub_en_coder= None
        self.en_gate = None
        self.sub_en_projection = None  
        
        #self.sub_ko_coder= None
        #self.ko_gate = None
        #self.sub_ko_projection = None
        self.target_ox_projection = None

        ### Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout

        self.ek_encoder = nn.LSTM(embed_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.ek_decoder = nn.LSTMCell(embed_size+self.hidden_size, self.hidden_size)
        self.ek_h_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)   
        self.ek_c_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ek_att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ek_combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        self.ek_target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.vocs), bias=False)        
                
        self.ke_encoder = nn.LSTM(embed_size, self.hidden_size, num_layers=2, bidirectional=True)
        self.ke_decoder = nn.LSTMCell(embed_size+self.hidden_size, self.hidden_size)
        self.ke_h_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)   
        self.ke_c_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ke_att_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False) 
        self.ke_combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size, bias=False)
        self.ke_target_vocab_projection = nn.Linear(self.hidden_size, len(self.vocab.vocs), bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        """       
        self.sub_encoder= nn.LSTM(embed_size, self.hidden_size, bidirectional=True)
        self.en_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_en_projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)  
        
        self.sub_decoder= nn.LSTM(embed_size, self.hidden_size)
        self.de_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_de_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        """
        self.sub_en_coder= nn.LSTM(embed_size, self.hidden_size)
        self.en_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_en_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        """
        self.sub_ko_coder= nn.LSTM(embed_size, self.hidden_size)
        self.ko_gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sub_ko_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False) 
        """      
        self.target_ox_projection = nn.Linear(self.hidden_size, 4, bias=False)        


    def forward(self, source: List[List[str]], target: List[List[str]], slang='en', tlang='ko') -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each ex0ample in the input batch. Here b = batch size.
        """
        
        ####################################################################       
        source_padded, source_lengths = self.parallel_encode_new(source, lang = slang)
        if slang == 'en':         
            enc_hiddens, dec_init_state = self.en2_encode(source_padded, source_lengths)             
        else:
            enc_hiddens, dec_init_state = self.ko_encode(source_padded, source_lengths)   
        """
        source_lengths = [len(s) for s in source]
        # Convert list of lists into tensors
        source_padded = self.vocab.vocs.to_input_tensor(source, device=self.device) 
        source_embedded = self.model_embeddings.vocabs(source_padded) 
        enc_hiddens, dec_init_state = self.en_encode(source_embedded, source_lengths)
        """            
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
                               
        tgt = [[w for w in s if w not in self.sbol] for s in target]
        #tgt = target
        parallel_padded = self.vocab.vocs.to_input_tensor(target, device=self.device)  #Tensor: (tgt_len, b)
        target_padded = self.vocab.vocs.to_input_tensor(tgt, device=self.device) 
        target_embedded, XO = self.parallel_decode_new(target, parallel_padded, lang=tlang)    #parallel_padded[:-1]) 

        if tlang == 'ko':  
            combined_outputs = self.ek_decode(enc_hiddens, enc_masks, dec_init_state, target_embedded) 
            P = F.log_softmax(self.ek_target_vocab_projection(combined_outputs), dim=-1)
        else:
            combined_outputs = self.ke2_decode(enc_hiddens, enc_masks, dec_init_state, target_embedded) 
            P = F.log_softmax(self.ke_target_vocab_projection(combined_outputs), dim=-1)

        target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]              
        P2 = F.log_softmax(self.target_ox_projection(combined_outputs), dim=-1)
        target_ox_log_prob = torch.gather(P2, index=XO[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0) + self.xo_weight * target_ox_log_prob.sum(dim=0)

        """    
        else:
            target_padded = self.vocab.vocs.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
            combined_outputs = self.ke_decode(enc_hiddens, enc_masks, dec_init_state, target_padded) 
            P = F.log_softmax(self.ke_target_vocab_projection(combined_outputs), dim=-1)
            target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
            target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:] 
            scores = target_gold_words_log_prob.sum(dim=0)
        """    
        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by theX `self.decode()` function.

        # Zero out, probabilities for which we have nothing in the target text
        #target_masks = (target_padded != self.vocab.vocs['<pad>']).float()
        
        # Compute log probability of generating true target words
        #target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
           
        return scores

    def parallel_encode(self,source):

        if type(source[0]) is not list:
           source = [source]       
    
        sbol = [['(', ')', ',', "'", '"'],'_']
        source_lengths, Z, Z_sub = get_sents_lenth(source,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in source]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)           
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        src_padded = self.vocab.vocs.to_input_tensor(source, device=self.device)  
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(src_padded,1,-1))]))     #각 문장으로 자른 뒤 문장내 어절 단위로 자른다 

        Z_flat = list(chain(*Z_sub))
        X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        X = pad_sequence(X).squeeze(-1)
        #X = torch.tensor(X, dtype=torch.float, device = self.device)
         
        X_embed = self.model_embeddings.vocabs(X)
        
        out,(last_h1,last_c1) = self.sub_encoder(X_embed)
        X_proj = self.sub_en_projection(out)
        X_gate = torch.sigmoid(self.en_gate(X_embed))
        X_way = self.dropout(X_gate * X_embed + (1-X_gate) * X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성         
        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
          torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩
        source_padded = pad_sequence(X_input).squeeze(-2)
        source_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return source_padded, source_lengths

    def parallel_encode_new(self,source, lang):

        if type(source[0]) is not list:
           source = [source]       
    
        source_lengths, Z, Z_sub = get_sents_lenth_new(source,self.sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in source]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)           
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        src_padded = self.vocab.vocs.to_input_tensor(source, device=self.device)  
        for i in range(len(source)):
            if len(src_padded[:,i]) != sum(XX[i]):
                print("s_len : {}, padded : {}, sumXX : {}".format(s_len[i],len(src_padded[:,i]), sum(XX[i])))
                print("XX : {}".format(XX[i]))
                print(source[i])

        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(
            torch.split(src_padded,1,-1))]))     #각 문장(batch)으로 자른 뒤 문장내 어절 단위로 자른다 

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        X = pad_sequence(X).squeeze(-1)
        #X = torch.tensor(X, dtype=torch.float, device = self.device)
         
        X_embed = self.model_embeddings.vocabs(X)
        #if lang == 'en':
        out,(last_h1,last_c1) = self.sub_en_coder(X_embed)
        X_proj = self.sub_en_projection(out[1:])               #sbol 부분 제거
        X_gate = torch.sigmoid(self.en_gate(X_embed[1:]))

        #else:
        #    out,(last_h1,last_c1) = self.sub_ko_coder(X_embed)
        #    X_proj = self.sub_ko_projection(out[1:])               #sbol 부분 제거
        #    X_gate = torch.sigmoid(self.ko_gate(X_embed[1:]))
        #    #print("gate embed proj : {} {} {}".format(X_gate.size(), X_embed.size(), X_proj.size()))

        X_way = self.dropout(X_gate * X_embed[1:] + (1-X_gate) * X_proj)       

        #문장단위로 자르고 어절 단위로 자른 뒤 각 어절의 길이만 남기고 나머지는 버린 후 연결 (cat) 하여 문장으로 재구성         
        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
          torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        
        # 재구성된 문장의 길이가 다르기 때문에 패딩
        source_padded = pad_sequence(X_input).squeeze(-2)
        source_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return source_padded, source_lengths

    def parallel_decode_old(self,target, tgt_padded):
    
        sbol = [['(', ')', ',', "'", '"'],'_']
        if type(target[0]) is not list: target = [target]
        target_lengths, Z, Z_sub = get_sents_lenth(target,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in target]  # 원래의 문장 길이
 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        
        max_l = max(s_len)  
        Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)]         
        XX =  [s+[max_l-s_len[i]-1] if max_l-1>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]

        X = pad_sequence(X).squeeze(-1)
        X = torch.tensor(X,device = self.device)
 
        X_embed = self.model_embeddings.target(X)
 
        X_gate = torch.sigmoid(self.gate(X_embed))
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)
        X_proj = self.dropout(torch.tanh(self.sub_de_projection(out)))
        X_way = X_gate * X_embed + (1-X_gate) * X_proj       
        
        #X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
        #  torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        X_input = [torch.cat([ss[:Z[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_padded = pad_sequence(X_input).squeeze(-2)
        #target_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return target_padded


    def parallel_decode(self,target, tgt,tgt_padded):
    
        sbol = [['(', ')', ',', "'", '"','_'],'_']
        if type(target[0]) is not list: target = [target]
        Z, XO = get_sents_lenth4(target,sbol) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in tgt]
        max_l = max(s_len)  
        Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)] 
        Z = [[sx for sx in s if sx>0] for s in Z] 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        max_l = max_l - 1
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device) 
        """
        print(Z_len, tgt_padded.shape)
        for i in range(4):
            print(XX[i], sum(XX[i]))
            print(XO[i])
        """
 
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))

        #Z_flat = list(chain(*Z_sub))
        #X = [s[:Z_flat[i]]for i,s in enumerate(X)]
        #X = torch.tensor(pad_sequence(X).squeeze(-1), dtype=torch.float, device=self.device)

        #X = torch.tensor(pad_sequence(X).squeeze(-1),device = self.device)
        X = pad_sequence(X).squeeze(-1)
 
        X_embed = self.model_embeddings.vocabs(X)        
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)
        #X_proj = torch.tanh(self.sub_de_projection(out))
        X_proj = self.sub_de_projection(out)
        #X_gate = torch.sigmoid(self.gate(torch.cat((X_embed,X_proj),-1)))
        X_gate = torch.sigmoid(self.de_gate(X_embed))
        X_way = self.dropout(X_gate * X_embed + (1-X_gate) * X_proj)      
        
        #X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
        #  torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
        X_input = [torch.cat([ss[:Z[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_padded = pad_sequence(X_input).squeeze(-2)
        XO = [torch.tensor(x) for x in XO]
        #XO = pad_sequence(XO)
        XO = torch.tensor(pad_sequence(XO),device = self.device)
        #target_lengths = [sum([wl for wl in s]) for s in Z_sub]
        
        return target_padded, XO


    def parallel_decode_new(self,target, tgt_padded, lang):
    
        sbols = {'_':1,'^':2,'`':3}
        if type(target[0]) is not list: target = [target]
        Z, XO, Z_sub = get_sents_lenth4_new(target,sbols) # Z:각 sentence 내의 각 어절의 길이로 구성  list[list]
        s_len = [len(s) for s in target]
        max_l = max(s_len)
        #Z = [s if max_l>s_len[i] else s[:-1]+[s[-1]-1] for i,s in enumerate(Z)] 
        #Z = [[sx for sx in s if sx>0] for s in Z] 
        max_Z = max(chain(*Z))  # 최대로 긴 어절
        Z_len = [len(s) for s in Z]    # 문장의 어절 갯수
        #max_l = max_l - 1
        XX =  [s+[max_l-s_len[i]] if max_l>s_len[i] else s for i,s in enumerate(Z)] # total(interval lenth) to be source lenth 
        """
        print(target[1])
        print(tgt[1])
        print("max_l :{}, tgt_padded :{}".format(max_l, tgt_padded.size()))
        print("Z_len : {}".format(Z_len))
        print("XX : {}".format(XX))
        """
        X = list(chain(*[torch.split(sss,XX[i])[:Z_len[i]] for i,sss in enumerate(torch.split(tgt_padded,1,-1))]))
        #target_padded = [[w for j,w in s if target[i][j] not in sbols.keys()] for i,s in enumerate(tgt_padded)]]
        X = pad_sequence(X).squeeze(-1) 
        X_embed = self.model_embeddings.vocabs(X) 

        #if lang == 'en':
        out,(last_h1,last_c1) = self.sub_en_coder(X_embed)
        X_proj = self.sub_en_projection(out[1:])
        X_gate = torch.sigmoid(self.en_gate(X_embed[1:]))
        #else:
        #    out,(last_h1,last_c1) = self.sub_ko_coder(X_embed)
        #    X_proj = self.sub_ko_projection(out[1:])
        #    X_gate = torch.sigmoid(self.ko_gate(X_embed[1:]))

        X_way = self.dropout(X_gate * X_embed[1:] + (1-X_gate) * X_proj)      

        X_input = [torch.cat([ss[:Z_sub[i][j]]for j,ss in enumerate(
            torch.split(sss,1,1))],0) for i,sss in enumerate(torch.split(X_way,Z_len,1))]
          
        target_embedded = pad_sequence(X_input).squeeze(-2)[:-1]
        XO = [torch.tensor(x) for x in XO]
        XO = torch.tensor(pad_sequence(XO),device = self.device) #<=[:-1]
        
        return target_embedded, XO

 
    def parallel_beam_encode(self,target):

        if type(target[0]) is not list: target = [target]
        s_len = [len(s)-1 for s in target]
        X = self.vocab.tgt.to_input_tensor(target, device=self.device)
        #tgt_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  
        #print(target)
        #X = pad_sequence(target)
        #X = torch.tensor(X,device = self.device)
 
        X_embed = torch.tensor(self.model_embeddings.target(X), device=self.device)
        #print(X_embed.shape)
 
        out,(last_h1,last_c1) = self.sub_decoder(X_embed)

        xxp = torch.tensor(s_len, device=self.device).unsqueeze(0).unsqueeze(-1)
        out = torch.gather(out, index=xxp.expand(1,-1,X_embed.size(2)), dim=0)
        #print("X_emd size {} {} {} {}".format(X_embed.shape, len(s_len), s_len[:5], xxp.shape))
        X_proj = torch.tanh(self.sub_de_projection(out))
        X_embed = torch.gather(X_embed, index=xxp.expand(1,-1,X_embed.size(2)), dim=0)
        X_gate = torch.sigmoid(self.gate(X_embed))
        X_way = X_gate * X_embed + (1-X_gate) * X_proj       
        
        return X_way

    def parallel_beam_encode2(self, X, lang, init_vecs= None):

        #X_embed = torch.tensor(self.model_embeddings.target(X), device=self.device).unsqueeze(0)
        X_embed = self.model_embeddings.vocabs(X).unsqueeze(0)
        
        #if len(X_embed.shape) <3:X_embed.unsqueeze(0)
        """
        if init_vecs is not None:
            print("X_embed / init_vec :{}, {}".format(X_embed.shape, init_vecs[0].shape))
        else:
            print("X_embed : {}".format(X_embed.shape))
        """
        #if lang =='en':
        out,(h,c) = self.sub_en_coder(X_embed,init_vecs)
        #X_proj = torch.tanh(self.sub_de_projection(out))
        X_proj = self.sub_en_projection(out)
        X_gate = torch.sigmoid(self.en_gate(X_embed))
        """
        else:
            if init_vecs is not None and len(init_vecs[0].size())<3 or len(X_embed.size()) <3:
                print("X_embed.size(), init_vecs[0].size() : {} {}".format(X_embed.size(), init_vecs[0].size()))
            out,(h,c) = self.sub_ko_coder(X_embed,init_vecs)
            #X_proj = torch.tanh(self.sub_de_projection(out))
            X_proj = self.sub_ko_projection(out)
            X_gate = torch.sigmoid(self.ko_gate(X_embed))
        """
        X_way = (X_gate * X_embed + (1-X_gate) * X_proj).squeeze(0)
        #print("X_way : {}".format(X_way.shape))      
        
        return X_way, (h,c)

    def en_encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        
         
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.ek_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]
       
        last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h = self.ek_h_projection(last_h)
        last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c = self.ek_c_projection(last_c)

        dec_init_state = (last_h, last_c)

        return enc_hiddens, dec_init_state


    def en2_encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        enc_hiddens, dec_init_state = None, None
      
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.ek_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]
       
        last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h = self.ek_h_projection(last_h)
        last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c = self.ek_c_projection(last_c)

        dec_init_state = (last_h, last_c)

        return enc_hiddens, dec_init_state  


    def ko_encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        enc_hiddens, dec_init_state = None, None
      
        X = pack_padded_sequence(source_padded, source_lengths)
        out, (last_h,last_c) = self.ke_encoder(X)
        enc_hiddens = pad_packed_sequence(out, batch_first=True)[0]
       
        last_h = torch.cat((last_h[0,:],last_h[1,:]), -1)
        last_h = self.ke_h_projection(last_h)
        last_c = torch.cat((last_c[0,:],last_c[1,:]), -1)
        last_c = self.ke_c_projection(last_c)

        dec_init_state = (last_h, last_c)

        return enc_hiddens, dec_init_state    
    
    
    def ek_decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_embedded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        #target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        
        enc_hiddens_proj = self.ek_att_projection(enc_hiddens)
        #Y = self.model_embeddings.target(target_padded)
        Y = target_embedded # self.parallel_decode(target, target_padded)
        for sp in torch.split(Y,1):
            #if len(sp.squeeze().size()) != len(o_prev.size()):
            #    print(sp.squeeze().size(),o_prev.size())
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='ko')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs

    def ke2_decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_embedded: torch.Tensor) -> torch.Tensor:

        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []

        enc_hiddens_proj = self.ke_att_projection(enc_hiddens)
        Y = target_embedded # self.parallel_decode(target, target_padded)
        for sp in torch.split(Y,1):
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='en')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs


    def ke_decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:

        target_padded = target_padded[:-1]
        dec_state = dec_init_state
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        combined_outputs = []
        
        enc_hiddens_proj = self.ke_att_projection(enc_hiddens)
        Y = self.model_embeddings.vocabs(target_padded)
        for sp in torch.split(Y,1):
            #if len(sp.squeeze().size()) != len(o_prev.size()):
            #    print(sp.squeeze().size(),o_prev.size())
            Ybar_t = torch.cat((sp.squeeze(0), o_prev), -1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks, tlang='en')
            combined_outputs.append(o_t)
            o_prev = o_t
        
        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs    
        

    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor, tlang) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hsplt_len_flattenedidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        
        if tlang == 'en':
            dec_state = self.ke_decoder(Ybar_t, dec_state)
        else:
            dec_state = self.ek_decoder(Ybar_t, dec_state)
        
        e_t = torch.bmm(enc_hiddens_proj, dec_state[0].unsqueeze(-1)).squeeze(-1)
       
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.splt_len_flattened
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        
        alpha_t = torch.softmax(e_t, -1).unsqueeze(-2)
        a_t = torch.bmm(alpha_t, enc_hiddens).squeeze(-2)
        #print("a_t.shape = ", a_t.shape)
        #print("dec_state[0].shape = ", dec_state[0].shape)        
        
        u_t = torch.cat((a_t, dec_state[0]), -1)
        if tlang == 'en':
            v_t = self.ke_combined_output_projection(u_t)
        else:
            v_t = self.ek_combined_output_projection(u_t)
        O_t = self.dropout(torch.tanh(v_t))

        ### END YOUR CODEsplt_len_flattened

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def ek_beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70, tlang:str='en') -> List[Khypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        #source_padded, source_lengths = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]

        slang = 'en' if tlang == 'ko' else 'ko'

        sbolX = [torch.tensor([self.vocab.vocs[w]], dtype=torch.long, device=self.device) for w in self.sbol]      
        sbol_init = ['']+[self.parallel_beam_encode2(sb,lang=tlang) for sb in sbolX]

        src_sents_var, src_len = self.parallel_encode_new(src_sent, slang) 
        if slang =='en':
            src_encodings, dec_init_vec = self.en2_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.ek_att_projection(src_encodings)
        else:
            src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.ke_att_projection(src_encodings)

        """
        src_sents_var = self.vocab.vocs.to_input_tensor([src_sent], self.device)e
        #src_encodings, dec_init_vec = self.encode(src_sents_var, [src_len])
        src_encodings_att_linear = self.ek_att_projection(src_encodings)
        """
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        hypotheses = [['<s>']]
        xhypotheses = [[1]]  ###???????
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))  # shape (b, src_len, h*2)

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            #y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            #y_t_embed = self.model_embeddings.target(y_tm1)                                # 수정된 부분
            if t<2:
                y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)       
                y_t_embed,next_init_vecs = self.parallel_beam_encode2(y_tm1, tlang, init_vecs=sbol_init[1][1]) # '<s>' 앞의  '_' 의 초기화 
            else:
                #####xxo = torch.tensor(prev_xos, dtype=torch.float,device=self.device)[:,-1].unsqueeze(0).unsqueeze(-1) #shape(1,batch,1)
                #print( "prev_init_vecs[0][0].shape:{}".format( prev_init_vecs[0][0].shape))   
                
                prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) for vecs in prev_init_vecs]    # shape (h,c) = ((1,batch,h),(1,batch,h)
                #prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) * xxo for vecs in prev_init_vecs]
                #print( "prev_init_vecs[0].shape:{}, xxo.shape".format( prev_init_vecs[0].shape),xxo.shape)           
                y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
                y_t_embed, next_init_vecs = self.parallel_beam_encode2(y_tm1,tlang, init_vecs=prev_init_vecs)   # 수정된 부분

            #y_t_embed = self.parallel_beam_encode(y_tm1)

            #print(y_t_embed.shape, att_tm1.shape)

            #x = torch.cat([y_t_embed.squeeze(0), att_tm1], dim=-1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None, tlang=tlang)

            # log probabilities over target words
            if tlang == 'ko':
                log_p_t = F.log_softmax(self.ek_target_vocab_projection(att_t), dim=-1)   #att_t's shape = (batch,hidden)
            else:
                log_p_t = F.log_softmax(self.ke_target_vocab_projection(att_t), dim=-1)                 
            log_p2, xos = F.log_softmax(self.target_ox_projection(att_t), dim=-1).max(-1)
            
            log_p2 = log_p2.unsqueeze(1).expand_as(log_p_t) * self.xo_weight   # 추가된 부분
            #xos = xos.unsqueeze(0).unsqueeze(-1)  # 추가된 부분

            #print("vecs, xos : {} {}".format(next_init_vecs[0].shape, xos.shape))
            #next_init_vecs = [vecs * torch.tensor(xos,dtype=torch.float,device=self.device) for vecs in next_init_vecs]

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t+log_p2).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.vocs)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.vocs)

            next_init_vecs = [next_vecs.squeeze(0) for next_vecs in next_init_vecs]

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            prev_xos = []
            prev_init_vecs = [[],[]]

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores ):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.vocs.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                #print("len(hypotheses):{},len(xhypotheses):{}, prev_hyp_ids :{}, xos:{}".format(
                #    len(hypotheses),len(xhypotheses),prev_hyp_ids, xos))
                new_xo = xhypotheses[prev_hyp_id] + [xos[prev_hyp_id]]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Khypothesis(value=new_hyp_sent[1:-1], 
                                                           xo = new_xo[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)   # to select (h_tm1, att_tm1)'s items matching new set
                    new_hyp_scores.append(cand_new_hyp_score)
                    prev_xos.append(new_xo)
                    if new_xo[-1] < 1:
                        prev_init_vecs[0].append(next_init_vecs[0][prev_hyp_id].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                        prev_init_vecs[1].append(next_init_vecs[1][prev_hyp_id].unsqueeze(0))
                    else:
                        prev_init_vecs[0].append(sbol_init[new_xo[-1]][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                        prev_init_vecs[1].append(sbol_init[new_xo[-1]][1][1].squeeze(0))                            

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            xhypotheses = prev_xos
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Khypothesis(value=hypotheses[0][1:],
                                                   xo = prev_xos[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        #print([w for w in completed_hypotheses[0].value][:20])
        #print([w for w in completed_hypotheses[0].xo][:20])

        temp_h = []
        #pre_v = ''
        for i,v in enumerate(completed_hypotheses[0].value):
           if i>0 and completed_hypotheses[0].xo[i] == 0:
              temp_h += [v]
           #elif i>0 and pre_v not in self.token and v not in self.token and completed_hypotheses[0].xo[i] == 0:
           #   temp_h += ['_'] + [v]
           else:
              temp_h += [self.sbol[completed_hypotheses[0].xo[i]-1]]+[v]
           #pre_v = v
        completed_hypotheses[0] = Khypothesis(value=temp_h,
                                             xo =  completed_hypotheses[0].xo,
                                             score= completed_hypotheses[0].score)

        return completed_hypotheses

    
    def ke_beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Ehypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        src_sents_var, src_len = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]
 
        #src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        #src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)
        src_encodings_att_linear = self.ke_att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.vocabs(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None, tlang='en')

            # log probabilities over target words
            log_p_t = F.log_softmax(self.ke_target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.vocs)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.vocs)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.vocs.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Ehypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Ehypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses    

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        #return self.model_embeddings.source.weight.device
        return self.model_embeddings.vocabs.weight.device  ##################################


    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


    def greedy_search(self, src_sent: List[str], max_decoding_time_step: int=70, tlang:str='en') -> List[Khypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
 
        #source_padded, source_lengths = self.parallel_encode(src_sent) 

        #src_sents_var =  source_padded
        #src_len = source_lengths[0]
        slang = 'en' if tlang == 'ko' else 'ko'

        slen = [(i, len([k for k in s if k not in self.sbol])) for i,s in enumerate(src_sent)]
        slen = sorted(slen, key = lambda x:x[1], reverse=True)
        src_sent = [src_sent[s[0]] for s in slen]
        sent_order = [ss[0] for ss in sorted([(i,s[0]) for i,s in enumerate(slen)], key = lambda x:x[1])]
        #  1:3     2 1
        #  2:1  => 3 2
        #  3:2     1 3  =>sent_order = [2,3,1]

        sbolX = [torch.tensor([self.vocab.vocs[w]], dtype=torch.long, device=self.device) for w in self.sbol]      
        sbol_init = ['']+[self.parallel_beam_encode2(sb, tlang) for sb in sbolX]

        src_sents_var, src_len = self.parallel_encode_new(src_sent, slang) 
        if slang =='en':
            src_encodings, dec_init_vec = self.en2_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.ek_att_projection(src_encodings)
        else:
            src_encodings, dec_init_vec = self.ko_encode(src_sents_var, src_len)
            src_encodings_att_linear = self.ke_att_projection(src_encodings)

        """
        src_sents_var = self.vocab.vocs.to_input_tensor([src_sent], self.device)e
        #src_encodings, dec_init_vec = self.encode(src_sents_var, [src_len])
        src_encodings_att_linear = self.ek_att_projection(src_encodings)
        """
        h_tm1 = dec_init_vec
        b_size = src_encodings.size(0)
        att_tm1 = torch.zeros(b_size, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocs['</s>']

        #hypotheses = [['<s>'] * b_size]
        sents = [['<s>']] * b_size
        #print("sents : {}".format(len(sents)))
        xos = [[1]] * b_size  ### '1' 은 사용되지 않고 버려짐 xos[1:-1]
        #sents_id = [[[1]] * b_size]
        live_sent_ids = torch.arange(b_size)
        scores = torch.zeros(b_size, dtype=torch.float, device=self.device)
        #completed_hypotheses = []
        batch_done = [{} for i in range(b_size)]

        t = 0
        while len(live_sent_ids) > 0: # and t < max_decoding_time_step:
            t += 1
            #hyp_num = len(hypotheses)
            """
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))  # shape (b, src_len, h*2)

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            """
            #y_tm1 = torch.tensor([self.vocab.vocs[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            #y_t_embed = self.model_embeddings.target(y_tm1)                                # 수정된 부분
            if t<2:
                y_tm1 = torch.tensor([self.vocab.vocs[s[-1]] for s in sents], dtype=torch.long, device=self.device) #.squeeze(-1)      
                #print("y_tm1.size()  : {}, sents[0][-1]: {}".format(y_tm1.size(), sents[0][-1]))
                pre_init_vecs=[sb.expand(1,b_size,self.hidden_size).contiguous() for sb in sbol_init[1][1]]

                y_t_embed,next_init_vecs = self.parallel_beam_encode2(y_tm1, tlang, init_vecs=pre_init_vecs)
            else:
                #####xxo = torch.tensor(prev_xos, dtype=torch.float,device=self.device)[:,-1].unsqueeze(0).unsqueeze(-1) #shape(1,batch,1)
                #print( "prev_init_vecs[0][0].shape:{}".format( prev_init_vecs[0][0].shape))   
                if len(prev_init_vecs[0]) >1:
                    prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0).contiguous() for vecs in prev_init_vecs]    # shape (h,c) = ((1,batch,h),(1,batch,h)
                else:
                    prev_init_vecs = [vecs[0].unsqueeze(0).contiguous() for vecs in prev_init_vecs] 
                #prev_init_vecs = [torch.cat(vecs,0).unsqueeze(0) * xxo for vecs in prev_init_vecs]
                #print( "prev_init_vecs[0].shape:{}, xxo.shape".format( prev_init_vecs[0].shape),xxo.shape)           
                y_tm1 = torch.tensor([self.vocab.vocs[s[-1]] for s in sents], dtype=torch.long, device=self.device)
                y_t_embed, next_init_vecs = self.parallel_beam_encode2(y_tm1,tlang, init_vecs=prev_init_vecs)   # 수정된 부분

            #y_t_embed = self.parallel_beam_encode(y_tm1)

            #print(y_t_embed.shape, att_tm1.shape)

            #x = torch.cat([y_t_embed.squeeze(0), att_tm1], dim=-1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear, enc_masks=None, tlang=tlang)

            # log probabilities over target words
            if tlang == 'ko':
                log_p_t, wid = F.log_softmax(self.ek_target_vocab_projection(att_t), dim=-1).max(-1)   #att_t's shape = (batch,hidden)
            else:
                log_p_t, wid = F.log_softmax(self.ke_target_vocab_projection(att_t), dim=-1).max(-1)  

            log_p2, xo = F.log_softmax(self.target_ox_projection(att_t), dim=-1).max(-1)  
            log_p2 = log_p2 * self.xo_weight          
            #log_p2 = log_p2.unsqueeze(1).expand_as(log_p_t) * self.xo_weight   # 추가된 부분
            #xos = xos.unsqueeze(0).unsqueeze(-1)  # 추가된 부분

            #print("vecs, xos : {} {}".format(next_init_vecs[0].shape, xos.shape))
            #next_init_vecs = [vecs * torch.tensor(xos,dtype=torch.float,device=self.device) for vecs in next_init_vecs]

            #live_hyp_num = b_size - len(batch_done)
            scores = (scores + log_p_t+log_p2)
            #contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t+log_p2).view(-1)
            # top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            #prev_hyp_ids = list(range(len(wid))) #top_cand_hyp_pos / len(self.vocab.vocs)
            #hyp_word_ids = wid #top_cand_hyp_pos % len(self.vocab.vocs)
            next_init_vecs = [next_vecs.squeeze(0) for next_vecs in next_init_vecs]
            sents  = [sents[i]+[self.vocab.vocs.id2word[wid[i].item()]] for i in range(len(live_sent_ids))]
            #print("b_size :{},len(xos) : {}, wid.size() : {}, xo.size() : {}".format(b_size,len(xos),wid.size(), xo.size()))
            xos = [xos[i] + [xo[i].item()] for i in range(len(live_sent_ids)) ]

            if eos_id not in wid and t < max_decoding_time_step:
                for i in range(len(live_sent_ids)):
                    if xo[i].item() < 1:
                        prev_init_vecs[0].append(next_init_vecs[0][i].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                        prev_init_vecs[1].append(next_init_vecs[1][i].unsqueeze(0))
                    else:
                        prev_init_vecs[0].append(sbol_init[xo[i].item()][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                        prev_init_vecs[1].append(sbol_init[xo[i].item()][1][1].squeeze(0)) 
                h_tm1 = (h_t, cell_t)
                att_tm1 = att_t

            else:
                prev_init_vecs = [[],[]]
                new_live_sent_ids = []
                new_temp_ids = []
                new_sents = []
                new_xos = []

                for i in range(len(live_sent_ids)):
                    b_id = live_sent_ids[i]
                    if wid[i].item() == eos_id or t == max_decoding_time_step:                       
                        batch_done[b_id] = {'sent':sents[i][1:-1], 'xo':xos[i][1:-1], 'score':scores[i].item()}
                    else:
                        new_live_sent_ids.append(b_id)
                        new_temp_ids.append(i)
                        new_sents.append(sents[i])
                        new_xos.append(xos[i])

                        if xo[i].item() < 1:
                            prev_init_vecs[0].append(next_init_vecs[0][i].unsqueeze(0))  #add batch array shape(hidden) = > (1,hidden) 
                            prev_init_vecs[1].append(next_init_vecs[1][i].unsqueeze(0))
                        else:
                            prev_init_vecs[0].append(sbol_init[xo[i].item()][1][0].squeeze(0))    #  remove direction shape(1,1,hidden) =>(1,hidden)
                            prev_init_vecs[1].append(sbol_init[xo[i].item()][1][1].squeeze(0)) 

                live_ids = torch.tensor(new_temp_ids, dtype=torch.long, device=self.device)
                h_tm1 = (h_t[live_ids], cell_t[live_ids])
                att_tm1 = att_t[live_ids]
                src_encodings = src_encodings[live_ids]
                src_encodings_att_linear = src_encodings_att_linear[live_ids]
                live_sent_ids = new_live_sent_ids
                sents = new_sents
                xos = new_xos
                scores = scores[live_ids]

            if len(live_sent_ids) == 0:
                break

        sents = []
        scores = []
        #print(len(batch_done),batch_done[1])
        for b in batch_done:

            sent = b['sent']
            xo = b['xo']

            X = []
            #pre_v = ''
            for i,v in enumerate(sent):
                if xo[i] == 0:
                    X += [v]
                else:
                    X += [self.sbol[xo[i]-1]]+[v]
            sents.append(' '.join(X))
            scores.append(b['score'])

        sents = [sents[i] for i in sent_order]
        scores = [scores[i] for i in sent_order]
        #print(sents[0])

        return sents, scores
