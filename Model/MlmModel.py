import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.LayerOne.EmbedPostional import EmbedPostional
from Model.EncoderBlock import EncoderBlock

class MlmModel(nn.Module):
    """Masked Language-model network phase 1 pre-training
    ids -> EmbeddingPositional -> N*EncoderBlock(Pre-norm) -> FinLayerNorm -> mlm head -> logits
    The head is tied to the token embedding layer, so the output logits are of shape (batch_size, seq_len, vocab_size)
    Args:
        vocab_size: size of the vocabulary
        d_model: dimension of the model
        nhead: number of heads
        num_layers: number of encoder blocks
        dropout: dropout rate
        max_seq_len: length of the sinusoidal positional table
        pad_token_id: id of the padding token
        token_emb_attr : name of the nn.Embedding attribute inside the EmbedPositional
    """

    def __init__(self,vocab_size:int,d_model:int,nhead:int,num_layers:int,dropout:float=0.1,max_seq_len:int,pad_token_id:int=0,token_emb_attr:str="token_emb"):
        super().__init__()
        self.pad_id = pad_token_id
        #Assumption : EmbedPositional(vocb_size,d_model,max_seq_len,dropout,pad_token_id) has an attribute called token_emb_attr which is the nn.Embedding layer for the tokens
        self.embed = EmbedPostional(vocab_size,d_model,max_seq_len,dropout,pad_token_id)
        self.blocks = nn.ModuleList([EncoderBlock(d_model,nhead,dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model) #Needed because the block is pre-norm, so the final output needs to be normalized before the mlm head
        self.head = nn.Linear(d_model,vocab_size,bias=False)
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))
        emb_weight = getattr(self.embed,token_emb_attr).weight
        assert emb_weight == (vocab_size,d_model), f"Embedding weight shape {emb_weight} does not match expected shape {(vocab_size,d_model)}"
        self.head.weight = emb_weight #tie the weights of the head to the embedding layer

    def encode(self,ids:torch.Tensor) -> torch.Tensor:
        """Encode the input ids into hidden representations
        Args:
            ids: input tensor of shape (batch_size, seq_len)
        Returns:
            output tensor of shape (batch_size, seq_len, d_model)
        """
        padding_mask = (ids != self.pad_id).to(ids.device) #True for non-pad tokens
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x,padding_mask)
        x = self.final_norm(x)
        return x

    def forward(self,ids:torch.Tensor,labels:torch.Tensor| None = None) -> torch.Tensor:
        h = self.encode(ids) #(batch_size, seq_len, d_model)
        if labels is not None:
            return F.linear(h,self.head.weight,bias=self.head_bias) #(batch_size, seq_len, vocab_size)
        # full logits for inference
        sel = labels != -100
        logits = F.linear(h,self.head.weight,bias=self.head_bias) #(batch_size, seq_len, vocab_size)
        targets = labels[sel]
        loss = F.cross_entropy(logits[sel],targets)
        return loss,logits,targets