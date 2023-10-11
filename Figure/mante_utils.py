import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

input_shape=4
output_shape=1
T=500
zero_time1=int(T/10)
zero_time2=int(T/2)
input_time=T-zero_time1-zero_time2

def rank(model):
    l=model.l
    pin=model.pin
    pout=model.pout
    gamma=torch.sum(torch.abs(l),0)/torch.sum(torch.norm(pin,2,0)+torch.norm(pout,2,0),0)
    tau=gamma*torch.norm(pin,2,0)+gamma*torch.norm(pout,2,0)+torch.abs(l)
    return tau.detach().cpu()

def train(rnn,targets, inputs,opt):
    
    rnn.to(device)
    outputs = rnn(inputs)
    criterion = torch.nn.MSELoss(reduction='sum')
    criterion.to(device)
    loss=torch.sqrt(criterion(outputs, targets))
    opt.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=20, norm_type=2)
    opt.step()
    return loss.item()



def generate_inputs():
    target_zero=torch.zeros(size=(1,1)).repeat(zero_time1+input_time,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,input_shape-2,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,input_shape-2,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,input_shape-2,1)).to(device)+offset
    signal=torch.concat((zeros1,x,zeros2))
    target=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(T,2,1)).to(device)
    if np.random.rand()<=0.5:
        target=target[0]
        context[:,0]=1
    else:
        target=target[1]
        context[:,1]=1
    target=target.repeat(T-zero_time1-input_time,1,1).to(device).to(torch.float)
    targets=torch.concat((target_zero,target))
    input_data=torch.concat((signal,context),axis=1).to(torch.float)
    return input_data,targets

def generate_batch(batchsize):
    target_zero=torch.zeros(size=(1,1)).repeat(zero_time1+input_time,batchsize,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,batchsize,input_shape-2,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,batchsize,input_shape-2,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(batchsize,2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,batchsize,input_shape-2,1)).to(device)+offset
    signal=torch.concat((zeros1,x,zeros2))
    labels=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(T,batchsize,2,1)).to(device)
    label=[]
    targets=[]
    for i in range(batchsize):
        if np.random.rand()<=0.5:
            label=labels[i,0]
            context[:,i,0]=1
        else:
            label=labels[i,1]
            context[:,i,1]=1
        targets.append(label.repeat(T-zero_time1-input_time,1,1,1).to(device).to(torch.float))
    targets=torch.concat(targets,dim=1)
    targets=torch.concat((target_zero,targets))
    input_data=torch.concat((signal,context),axis=2).to(torch.float)
    return input_data,targets


def generate_inputs_offset():
    target_zero=torch.zeros(size=(1,1)).repeat(zero_time1+input_time,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,input_shape-2,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,input_shape-2,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,input_shape-2,1)).to(device)+offset
    signal=torch.concat((zeros1,x,zeros2))
    target=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(T,2,1)).to(device)
    if np.random.rand()<=0.5:
        target=target[0]
        context[:,0]=1
    else:
        target=target[1]
        context[:,1]=1
    target=target.repeat(zero_time2,1,1).to(device).to(torch.float)
    targets=torch.concat((target_zero,target))
    input_data=torch.concat((signal,context),axis=1).to(torch.float)
    return input_data,targets,torch.concat((zeros1,x-offset,zeros2)),offset


def generate_context_switch():
    target_zero=torch.zeros(size=(1,1)).repeat(T-zero_time2,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,input_shape-2,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,input_shape-2,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,input_shape-2,1)).to(device)+offset
    signal=torch.concat((zeros1,x,zeros2)).repeat(2,1,1)
    target=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(2*T,2,1)).to(device)
    if np.random.rand()<=0.5:
        target1=target[0]
        context[:T,0]=1
        target2=target[1]
        context[T:,1]=1
    else:
        target1=target[1]
        context[:T,1]=1
        target2=target[0]
        context[T:,0]=1
    target1=target1.repeat(zero_time2,1,1).to(device).to(torch.float)
    targets1=torch.concat((target_zero,target1))
    target2=target2.repeat(zero_time2,1,1).to(device).to(torch.float)
    targets2=torch.concat((target_zero,target2))
    targets=torch.concat((targets1,targets2),)
    input_data=torch.concat((signal,context),axis=1).to(torch.float)
    return input_data,targets

def generate_batch_switch(batchsize,D=200,P=0.1):
    #d=np.random.randint(100,D)
    d=0
    if np.random.rand()<=P:
        d=D
    target_zero=torch.zeros(size=(1,1)).repeat(zero_time1+input_time,batchsize,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,batchsize,input_shape-2,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,batchsize,input_shape-2,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(batchsize,2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,batchsize,input_shape-2,1)).to(device)+offset
    signal=torch.concat((zeros1,x,zeros2))
    labels=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(T,batchsize,2,1)).to(device)
    label=[]
    targets=[]
    for i in range(batchsize):
        if np.random.rand()<=0.5:
            label=labels[i,0]
            context[:,i,0]=1
            j=1
        else:
            label=labels[i,1]
            context[:,i,1]=1
            j=0
        targets.append(torch.concat([label.repeat(T-zero_time1-input_time-d,1,1,1).to(device).to(torch.float),labels[i,j].repeat(d,1,1,1).to(device).to(torch.float)]))
    targets=torch.concat(targets,dim=1)
    targets=torch.concat((target_zero,targets))
    input_data=torch.concat((signal,context),axis=2).to(torch.float)
    input_data[-d:,:,2:]=input_data[-d:,:,2:].flip(2)
    return input_data,targets

def generate_random_batch_switch(batchsize,D=200,P=0.1):
    Input_data,Targets=generate_batch_switch(1,D,P)
    for i in range(batchsize-1):
        input_data,targets=generate_batch_switch(1,D,P)
        Input_data=torch.cat((Input_data,input_data),dim=1)
        Targets=torch.cat((Targets,targets),dim=1)
    return Input_data,Targets

from sympy.matrices import Matrix, GramSchmidt


def orthogo_tensor(x):
    m, n = x.size()
    x_np = x.t().numpy()
    matrix = [Matrix(col) for col in x_np.T]
    gram = GramSchmidt(matrix)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))
        ort_list.append(vector)
    ort_list = np.mat(ort_list)
    ort_list = torch.from_numpy(ort_list)
    ort_list = F.normalize(ort_list,dim=1)
    return ort_list.float()

def get_coff(model):
        r=model.R.cpu()
        pin=model.pin.cpu().detach()
        Win=model.Win.cpu().detach()
        GS=torch.concat([Win,pin],dim=1)
        GS=orthogo_tensor(GS.T).T
        Win,pin=torch.split_with_sizes(GS,[4,GS.shape[1]-4],1)
        beta=torch.matmul(pin.T,pin).inverse().matmul(torch.matmul(pin.T,Win))
        Wv=Win-torch.matmul(pin,beta)
        v=torch.matmul(Wv.T,Wv).inverse().matmul(torch.matmul(Wv.T,r))
        mu=torch.matmul(pin.T,pin).inverse().matmul(torch.matmul(pin.T,r))
        return mu