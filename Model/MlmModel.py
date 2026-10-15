import torch
import torch.nn as nn
import torch.nn.functional as F

from EncoderBlock import EncoderBlock

class MlmModel(nn.Module):
    def __init__(self,vocab_size,d_model,n_layer,dropout,max_len,pad_id,tok_emb_arr="tok_emb"):
        super().__init__()
        self.encoder = EncoderBlock(vocab_size,d_model,n_layer,dropout,max_len,pad_id,tok_emb_arr)
        self.head = nn.Linear(d_model, vocab_size,bias=False)
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))
        emb_weight = self.encoder.embed.tok_emb.weight()
        assert emb_weight.shape[0] == vocab_size, "Embedding weight shape does not match vocab size"
        print(f"embedding weight shape: {emb_weight.shape}, vocab size: {vocab_size}")
        self.head.weight = emb_weight
        self.head.bias = self.head_bias #Weights are tied to the embedding layer

    def encode(self,ids,padding_mask=None):
        return self.encoder(ids,padding_mask)

    def forward(self,ids,labels=None,padding_mask=None):
        h = self.encode(ids,padding_mask)
        if labels is not None:
            return F.linear(h,self.head.weight,bias=self.head.bias)
        sel = labels != -100
        logits = F.linear(h,self.head.weight,bias=self.head.bias)
        targets = labels[sel]
        loss = F.cross_entropy(logits[sel],targets)
        return loss,logits,targets