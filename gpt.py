import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'


block_size = 64
batch_size = 128
max_iters = 5000
learning_rate = 3e-4
eval_iters=250
n_embd = 384
n_head = 8
n_layer=8
dropout =0.2

from datasets import load_dataset
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

chars = set()
for i, example in enumerate(dataset):
    chars.update(example["text"])
    if i > 5000:
        break
chars = sorted(list(chars))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")


string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int.get(c, 0) for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])


text_data = []
for i, example in enumerate(dataset):
    text_data.append(example["text"])
    if i > 3000:
        break

text = "".join(text_data)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

n = int(0.8*len(data))
train_data=data[:n]
val_data = data[n:]


def get_batch(split):
  data= train_data if split =='train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x=torch.stack([data[i:i+block_size] for i  in ix ])
  y=torch.stack([data[i+1:i+block_size+1] for i  in ix])
  x,y = x.to(device),y.to(device)
  return x,y

class Head(nn.Module):
  def __init__(self, head_size):
    super() .__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key (x)
    q = self.query (x)
    wei = q @ k.transpose(-2,-1) * k.shape[-1] **- 0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim =- 1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads,n_embd)
    self.dropout =nn.Dropout(dropout)
  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads],dim=-1)
    return out

class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net=nn.Sequential(
        nn.Linear(n_embd, 4* n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd,n_embd),
        nn.Dropout(dropout),
    )
  def forward(self,x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    head_size=n_embd// n_head
    self.sa=MultiHeadAttention(n_head,head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def forward(self,x):
    y= self.sa(x)
    x= self.ln1(x+y)
    y= self.ffwd(x)
    x= self.ln2(x+y)
    return x


class  GPTLanguageModel(nn.Module):
  def __init__(self,vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
      self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
      self.ln_f = nn.LayerNorm(n_embd)
      self.Lm_head = nn.Linear(n_embd, vocab_size)
      # Add position embedding table
      self.position_embedding_table = nn.Embedding(block_size, n_embd)


  def forward(self, index, targets=None) :
    B,T = index.shape
    tok_emb = self.token_embedding_table(index)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks (x)
    x = self.ln_f (x)
    logits = self.Lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, index, max_new_tokens):
    for _ in range(max_new_tokens):
      index_cond = index[:, -block_size:]
      logits, loss = self.forward(index_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim =- 1)
      index_next = torch.multinomial(probs, num_samples=1)
      index = torch.cat((index, index_next), dim=1)
    return index

model = GPTLanguageModel(vocab_size)
m = model.to(device)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_iters == 0:
    losses = estimate_loss()
    print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

  xb, yb = get_batch('train')

  logits, loss = model.forward(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(loss.item())

context = torch.zeros((1,1), dtype=torch.long,device=device)
generated_chars = decode(m.generate(context,max_new_tokens=500)[0].tolist())
print(generated_chars)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"The model has {num_params} parameters.")


model_path = "model_weights.pth"

torch.save(model.state_dict(), model_path)

print(f"Model weights saved to {model_path}")

import os


model_path = "model_weights.pth"


absolute_model_path = os.path.abspath(model_path)

print(f"The exact path of the model weights file is: {absolute_model_path}")