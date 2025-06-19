import torch 
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64 # how many independent sequences will be processed in parallel (batch_dimension)
block_size = 16 # what is the maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

print(f"Device in Use: {device}")
torch.manual_seed(1337)

# read the text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Tokenization (naive character level tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a dictionary to map character to integer (corresponding to the index in chars)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
# create encoder/decoder lambda functions
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train/Validation Split
data = torch.tensor(encode(text), dtype=torch.long)
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

# essentially a table of the likelihood of token B given token A, for the entire set of tokens in the vocabulary (predict the next token given the current token)
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and trargets are both (B,T) tensor of integers
        # B: # of batches (4), T: # of tokens per batch (8)
        logits = self.token_embedding_table(idx) # (B,T,C) -> C: channel (vocab size)

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
            # get the predictions
            logits, loss = self(idx) 
            # focus only on the last time step (since we just want to GENERATE the NEXT token, not validating or anything)
            logits = logits[:, -1, :] # become (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution (next token for each batch)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Run training step

for iter in range(max_iters):
    
    # evaluate loss on train and val sets every evaluation interval
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss and step the optimizer
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) 
    loss.backward()
    optimizer.step()

# generate from the model (inference)
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))