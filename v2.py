import torch 
import torch.nn as nn
from torch.nn import functional as F
from character_tokenizer import CharacterTokenizer
import matplotlib.pyplot as plt


# hyperparameters
batch_size = 64 # how many independent sequences will be processed in parallel (batch_dimension)
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

print(f"Device in Use: {device}")
torch.manual_seed(1337)

# Initialzie character tokenizers
tokenizer = CharacterTokenizer()
tokenizer.load_corpus('input.txt')

# Train/Validation Split
data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
n = int(0.9*len(data)) # number of tokens to be used for training (first 90%)
train_data = data[:n]
val_data = data[n:]

# Loading data
def get_batch(split):
    # generate a random, small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # switch dataset whether we want training or validation data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # get batch_size random 'cuts' in the dataset (starting indices/offsets for our chunks basically)
    x = torch.stack([data[i:i+block_size] for i in ix]) # generate the chunks based on the 'cuts' and the block_size -> do it for each of the batches and stack them together in one tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # run evaluation for a few iterations to get consistent average
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set model back to train mode
    return out

class FeedForward(nn.Module):
    # simple linear layer followed by non-linearity
    # layer size dependent on embedding size (n_embd), 
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    # one head of self-attention
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # apply key and query matrices
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        
        # compute attention scores ("affinties")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform weighted aggregation of the value
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    # mutliple heads of self-attention in parallel
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class Block(nn.Module):
    # Transformer Block: Communication followed by computation (attention -> aggregation)
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # total head size should be the same as embedding dimension
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
    
# essentially a table of the likelihood of token B given token A, for the entire set of tokens in the vocabulary (predict the next token given the current token)
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(tokenizer.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, tokenizer.vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and trargets are both (B,T) tensor of integers
        # B: # of batches (4), T: # of tokens per batch (8)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) -> C: channel (n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x =  tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply one head of self attention -> (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab size)

        if targets is None:
            loss=None
        else:
            # re-shape logits and targets to be (B*T, C) and (B*T, 1) respectively
            B , T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)    
            # measure loss using negative log-likelihood
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond) 
            # focus only on the last time step (since we just want to GENERATE the NEXT token, not validating or anything)
            logits = logits[:, -1, :] # become (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution (next token for each batch)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# For live plotting
train_losses = []
val_losses = []
steps = []
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Train Loss')
line2, = ax.plot([], [], label='Val Loss')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.legend()

# Run training step
for iter in range(max_iters):
    # evaluate loss on train and val sets every evaluation interval
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps.append(iter)
        line1.set_data(steps, train_losses)
        line2.set_data(steps, val_losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss and step the optimizer
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) 
    loss.backward()
    optimizer.step()

plt.ioff()
plt.show()

# generate from the model (inference)
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context,max_new_tokens=500)[0].tolist()))