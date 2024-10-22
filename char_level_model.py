import pandas as pd
import json
import torch
import torch.nn.functional as F
from helper_funcs import build_ngram_dataset
import numpy as np
from microtorch import Embedding,Linear,Flatten,Sequential, Tanh

with open('top_posts_data.json', 'r') as json_file:
    data = json.load(json_file)

first_comments = [data['data'][i]['node']['comments']['nodes'][0]['body'] for i in range(len(data['data']))]
corpus = ''.join(first_comments)

chars = sorted(list(set(corpus)))
chars.insert(0,'<start>')
chars.append('<end>')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

xs = []
ys = []

xs,ys = build_ngram_dataset(data = first_comments,stoi=stoi)
len(xs),len(ys)

x=torch.tensor(xs)
y=torch.tensor(ys)

indices = np.arange(len(x)-1)
np.random.seed(42)
np.random.shuffle(indices)

train_size = int(0.8 * (len(x)-1))
val_size = int(0.1 * (len(x)-1))
test_size = (len(x)-1) - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_xs, train_ys = x[train_indices], y[train_indices]
val_xs, val_ys = x[val_indices], y[val_indices]
test_xs, test_ys = x[test_indices], y[test_indices]

embed_dim = 20
hidden_dim = 300
vocab_size=len(itos)
block_size = 3

NN = Sequential([
    Embedding(vocab_size,embed_dim),
    Flatten(),
    Linear(embed_dim* block_size,hidden_dim),
    Tanh(),
    Linear(hidden_dim,hidden_dim),
    Tanh(),
    Linear(hidden_dim,vocab_size)
])

parameters= NN.parameters()

def train_model(model,iterations,Xtr,Ytr,batch_size,learning_rate,params):
    for i in range(iterations):    
        indexes = torch.randint(0,len(xs)-1,(batch_size,))
        for p in params:
            p.grad = None
        x_mini = Xtr[indexes]
        y_mini = Ytr[indexes]
        out = model(x_mini)
        loss = F.cross_entropy(out,y_mini)
        if (i%5000 == 0) or (i==iterations-1):
            print(f'{i}: , loss: {loss}')
        loss.backward()
        learning_rate = learning_rate if i<iterations/2 else learning_rate * 0.1
        for p in params:
            p.data += - learning_rate * p.grad

train_model(NN,200000,xs=train_xs,ys=train_ys,batch_size=64,learning_rate=0.1, params =parameters)

def generate_text(model, stoi, itos, start_token='<start>', end_token='<end>', max_length=400):
    context_window = block_size  
    
    start_idx = stoi[start_token]
    context = [start_idx] * context_window
    generated = []
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([context])
            logits = model(x)  
            probs = torch.softmax(logits, dim=1)
            
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            next_token = itos[next_token_idx]
            
            if next_token == end_token:
                break
            generated.append(next_token)
            
            context = context[1:] + [next_token_idx]
    
    return ''.join(generated)

# print(generate_text(model=NN,stoi=stoi,itos=itos))