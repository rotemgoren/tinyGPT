import torch
import wget
from models import  GPTModel
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iter=5000
eval_interval=500
lr=3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters =200
n_embed=384
num_heads = 6
num_layers = 6
dropout = 0.2

url =  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
filename=wget.download(url)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
# let's look at the first 1000 characters
#print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#print(encode("hii there"))
#print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
torch.manual_seed(1337)

# train_data[:block_size+1]
#
# x= train_data[:block_size]
# y= train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target {target}")

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x,y = x.to(device),y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out ={}
    model.eval()
    for split in ['train' ,  'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            _,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
#
# print('----')
#
# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")
#

torch.manual_seed(1337)

model = GPTModel(vocab_size,n_embed,block_size,num_heads,num_layers,dropout,device).to(device)
print(sum(p.numel() for p in model.parameters())/1e6 , 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

batch_size=32
for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"ster {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
context =torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(idx=context, max_new_tokens=100)[0].tolist()))


