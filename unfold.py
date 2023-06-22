import torch
import math,time
import random
import torch.nn.functional as F

class SimpleMarkovLM(torch.nn.Module):

    def __init__(self,k,dim,vocab_size):
        
        super(SimpleMarkovLM, self).__init__() 
        self.k = k-1
        self.dim = dim
        self.vocab_size = vocab_size
        # network layers
        self.embed = torch.nn.Embedding(vocab_size,dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
        self.conv1 = torch.nn.Conv1d(dim,dim,3,padding='same')
        self.conv2 = torch.nn.Conv1d(dim,dim,3,padding='same')
        self.conv3 = torch.nn.Conv1d(dim,dim,3,padding='same') 
        #self.transformer = torch.nn.TransformerEncoderLayer(dim,2,norm_first=True)
        self.mlp1 = torch.nn.Linear(dim,dim)
        self.mlp2 = torch.nn.Linear(dim,vocab_size)

    def forward(self,batch):
        x = self.embed(batch)
        L,batch_size,_ = x.shape  
        # take overlapping k-mer slices  
        x = x.unfold(0,self.k,1).reshape(-1,self.dim,self.k)
        # run through convolutional layers 
        x = F.gelu(self.norm1(self.conv1(x))) + x
        x = F.gelu(self.norm2(self.conv2(x))) + x
        x = F.gelu(self.norm3(self.conv3(x))) + x 
        #x = x.permute(2,0,1) # transformer expects (seq_len,batch_size,hidden_size)
        #x = self.transformer(x).permute(1,2,0) # restore original shape
        # pool and project 
        x = x.sum(-1) 
        x = x.reshape(L-self.k+1,batch_size,self.dim)
        output = F.log_softmax(self.mlp2(self.mlp1(x)),dim=-1)
        return output 

# hparams
k = 15
dim = 128
batch_size = 8
vocab_size = 5
N = 50

# build dataset
dataset = []
for i in range(N):
    L = random.randint(800,1200)
    a = torch.randint(0,vocab_size,(L,batch_size))      
    dataset.append(a)

net = SimpleMarkovLM(k,dim,vocab_size)
optimizer = torch.optim.Adam(net.parameters())
nll_loss = torch.nn.NLLLoss(reduction='mean')
running_loss = 0.0

for epoch in range(10):
    for i,x in enumerate(dataset): 
        s = time.time()
        y = net(x)
        e = time.time()
        #print(f'forward time: {e-s:.3f}')
        pred = y.argmax(dim=-1)
        # shifted target 
        pad_suffix = x.new_ones((1,x.shape[1]))
        tgt = torch.cat([x[k-1:,:] , pad_suffix],dim=0)
        # backprop 
        optimizer.zero_grad() 
        loss = nll_loss(y.reshape(-1,y.shape[-1]),tgt.reshape(-1))      
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i > 0 and i % 100 == 0:
            avg_loss = running_loss / 100
            print(f'epoch={epoch}, i={i}, loss={avg_loss}, ppl={math.exp(avg_loss)}')
            running_loss = 0.0

