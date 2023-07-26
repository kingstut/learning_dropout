import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt 
import numpy as np 
import random

max_steps = 10
batch_size = 256
n = batch_size
num = 3
exclude = False

#load the data
data =  datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
if exclude:
    data = list(filter(lambda i: i[1] == num, data))

train_loader = torch.utils.data.DataLoader(data,
    batch_size=batch_size, shuffle=False,)

train_data = torch.stack([torch.load(f'one_example/{i}.pt') for i in range(10)], dim=0)
test_data =  datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size, shuffle=True,)

# init
n_hidden = 100 # the number of neurons in the hidden layer of the MLP

seed = 214748364
g = torch.Generator().manual_seed(seed) # for reproducibility
# Layer 1
W1 = torch.randn((784, n_hidden), generator=g) 
b1 = torch.randn(n_hidden, generator=g) 

# Layer 2
W2 = torch.randn((n_hidden, 10), generator=g)
b2 = torch.randn(10, generator=g)

lr = 0.1
p1 = 0.4
p2 = 1
plot = True
s = f'one_eg_p1_{p1}_p2_{p2}_lr=0.1_seed_{seed}'
"""
W1 = torch.load(f'saved/tensor_0_{s}.pt')
b1 = torch.load(f'saved/tensor_1_{s}.pt')
W2 = torch.load(f'saved/tensor_2_{s}.pt')
b2 = torch.load(f'saved/tensor_3_{s}.pt')
"""
parameters = [W1, b1, W2, b2]
lossi = []

#test loss
def get_acc():
    test_loss = 0
    correct = 0
    for Xb, Yb in test_loader:
        xcat = Xb.view(Xb.shape[0], -1) # concatenate the vectors
        hpreact = xcat @ W1 + b1
        h = torch.relu(hpreact) 
        output = h @ W2 + b2
        
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(Yb.data.view_as(pred)).long().sum()


    test_loss /= len(test_loader.dataset)
    return 100.0 * correct / len(test_loader.dataset)

with torch.no_grad():
    for i in range(1, max_steps+1):
        lr = lr*(0.8)
        for batch_idx, (Xb, Yb) in enumerate(train_loader):
            #Xb = train_data
            #Yb = torch.tensor(range(10))
            #n = 10
            if Yb.shape[0]!=n:
                continue
            xcat = Xb.view(Xb.shape[0], -1) # concatenate the vectors
            hpreact = xcat @ W1 + b1
            h = torch.relu(hpreact)
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, Yb) 
            
            # backward pass
            dlogits = F.softmax(logits, 1)
            dlogits[range(n), Yb] -= 1
            dlogits /= n
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(0)
            dpreact = (h>0) * (dlogits @ W2.T)
            dW1 = xcat.T @ dpreact
            db1 = dpreact.sum(0)

            W1 += -lr * dW1 * torch.bernoulli(torch.ones_like(dW1)*p1)
            b1 += -lr * db1 * torch.bernoulli(torch.ones_like(db1)*p1)

            W2 += -lr * dW2 * torch.bernoulli(torch.ones_like(dW2)*p2)
            b2 += -lr * db2 * torch.bernoulli(torch.ones_like(db2)*p2)            

            # track stats
            lossi.append(loss.item())
            #print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        print("%.2f" % get_acc().item())

if plot:
    plt.plot(lossi)
    np.save(f'losses/{s}', lossi)
    plt.savefig(f'train_plots/{s}.png')

    for i, p in enumerate(parameters):
        torch.save(p, f'saved/tensor_{i}{s}.pt')
