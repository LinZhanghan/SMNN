import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])
data_path=r'F:\codingtime\data'
input_shape=784
output_shape=10
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)



def generate_input(loader):
    examples = enumerate(loader)
    batch_idx, (data, targets) = next(examples)
    data=data.reshape((-1,28,28)).to(device)
    targets=targets.to(device)
    targets_onehot=torch.nn.functional.one_hot(targets).float().reshape(targets.shape[0],-1,1).to(device)
    return data,targets_onehot,targets

def train(model,batch_size,opt):
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    model.to(device)
    data,targets_onehot,targets=generate_input(train_loader)
    output = model(data)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    loss=criterion(output, targets_onehot)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    opt.step()
    return loss.item()

def test(model,batch_size=1000):
    
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    test_losses=[]
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data,targets_onehot,targets=generate_input(test_loader)
        output= model(data)
        test_loss += F.cross_entropy(output, targets_onehot,reduction='sum').item()
        pred=output.data.max(1)[1]
        correct+=pred.eq(targets.data.view_as(pred)).sum()
            
    test_loss /= data.shape[0]
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, data.shape[0],100. * correct /data.shape[0]))
    return correct /data.shape[0]


def rank(model):
    l=model.l
    pin=model.pin
    pout=model.pout
    gamma=torch.sum(torch.abs(l),0)/torch.sum(torch.norm(pin,2,0)+torch.norm(pout,2,0),0)
    tau=gamma*torch.norm(pin,2,0)+gamma*torch.norm(pout,2,0)+torch.abs(l)
    return tau.detach().cpu()