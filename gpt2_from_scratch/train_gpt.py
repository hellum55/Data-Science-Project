import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel

#===============================================================================
#Attention mechanism
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #regularization
        self.n_head = config.n_head
        self.n_head = config.n_head
        #Add a bias or a mask to the attention mechanism
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, number of channels (in_embd)
        #calculate query, key, value for all heads in batch and move head forward to be the second dimension
        #nh is the number of heads, hs is the size of the head and c is the number of channels = nh*hs
        #example: GPT2 - 124m has 12 heads, 64 head size and 768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(batch, head, seq_length, head_features)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(batch, head, seq_length, head_features)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(batch, head, seq_length, head_features)
        #calculate the attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v #attention scores applied to the values
        y = y.transpose(1, 2).contiguous().view(B, T, C) #rearrange the dimensions
        #output projection
        y = self.c_proj(y)
        return y
    
#===============================================================================
#Define the MLP-block
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) #One linear layers
        self.gelu = nn.GELU(approximate='tanh') #Gaussian Error Linear Unit activation function
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) #One linear layers

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#===============================================================================
# Define the 'Block' class
class Block(nn.Module):
    #Initialize the class 
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    #Implement the forward pass
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

#===============================================================================
@dataclass
class GPTConfig:
    block_size: int = 1024 #maximum sequence length
    vocab_size: int = 50257 #vocabulary size = 50.000 BPE merges, 256 bytes tokens, 1 <jenoftext> token
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads
    n_embd: int = 768 #embedding dimension

#===============================================================================
# Define the GPT class
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), #weights of positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #blocks, h=hidden
            ln_f = nn.LayerNorm(config.n_embd) #final layer normalization
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #final layer and classifier

    def forward(self, idx):
        #idx is the shape of (B, T) where B is the batch size and T is the sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arrange(0, T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb = self.transformer['wpe'](pos) #shape (T, n_embd)
        tok_emb = self.transformer['wte'](idx) #shape (B, T, n_embd)
        x = tok_emb + pos_emb #shape (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer normalization and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #shape (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#===============================================================================
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

#prefic tkoens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, i'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
x = tokens.to('cuda')