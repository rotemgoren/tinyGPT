import torch
from torch import nn
from torch.nn import functional as F
import math

def new_gelu(x):
    return 0.5 * x * (1.0+torch.tanh(torch.tensor(math.sqrt(2.0/math.pi))) * (x + 0.044715 * torch.pow(x, 3.0)))

class AttentionHead(nn.Module):
    def __init__(self,head_size,n_embed,block_size,dropout):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size)
        self.query = nn.Linear(n_embed,head_size)
        self.value = nn.Linear(n_embed,head_size)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C =x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2,-1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        weights = F.softmax(weights,dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size,block_size,dropout):
        super().__init__()
        n_embed = head_size * num_heads
        self.heads = nn.ModuleList([AttentionHead(head_size,n_embed,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embed,dropout):
        super().__init__()

        self.fc = nn.Linear(n_embed,4 * n_embed)
        self.proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x  = self.fc(x)
        x = new_gelu(x)
        x = self.proj(x)
        x = self.dropout(x)

        return x

class LayerNorm1D:
    def __init__(self,dim,eps=1e-5,momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # self.running_mean = torch.zeros(dim) #for batchNorm
        # self.running_var = torch.ones(dim) #for batchNorm

    def __call__(self,x):
        ### for batchNorm ###
        # if self.training:
        #     xmean = x.mean(1,keepdim=True)
        #     xvar = x.var(1,keepdim=True)
        # else:
        #     xmean = self.running_mean
        #     xvar = self.running_var

        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)

        xhat = (x-xmean) / torch.sqrt(xvar + self.eps) #normaliz to unit variance
        self.out = self.gamma * xhat +self.beta

        ### for batchNorm ###
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum * xmean
        #         self.running_var = (1 - self.momentum)*self.running_var +self.momentum *xvar

        return self.out

    def parameters(self):
        return [self.gamma , self.beta
                ]
class TransformerBlock(nn.Module):
    def __init__(self,n_embed,num_heads,block_size,dropout):
        super().__init__()
        head_size = n_embed//num_heads
        self.selfattn = MultiHeadAttention(num_heads,head_size,block_size,dropout)
        self.ffwd = FeedForward(n_embed,dropout)

        self.ln1 = nn.LayerNorm(n_embed) #pre layer norm
        self.ln2 = nn.LayerNorm(n_embed) #pre layer norm


    def forward(self,x):
        x = x + self.selfattn(self.ln1(x)) #residual 1
        x = self.ln2(x)
        x = x + self.ffwd(x) #residual 2
        return x

class GPTModel(nn.Module):

    def __init__(self, vocab_size,n_embed,block_size,num_heads,num_layers,dropout,device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)


        self.position_embedding_table = nn.Embedding(block_size,n_embed)

        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, num_heads ,block_size,dropout) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=self.device)) #(T,C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc