import torch
import numpy as np
from model_settings import *

def generate_inputs():

    """

    Generate an input of model

    offset: noisy signals' offset
    signals: noisy signal
    input_signals: input signals in channel 1 and 2
    context: context signals in channel 3 and 4

    Returns:
        input_data: input data of model
        targets: target outputs of model

    """

    offset=torch.tensor(np.random.choice([-0.5,0.5],(2,1))).to(device)  
    signals=torch.randn(size=(T-zero_time1-zero_time2,input_shape,1)).to(device)+offset
    input_signals=torch.concat((zeros1,signals,zeros2))
    target=torch.where(offset>0,1,-1)
    context=torch.zeros(size=(T,2,1)).to(device)
    #choose contextual cue
    if np.random.rand()<=0.5:
        target=target[0]
        context[:,0]=1
    else:
        target=target[1]
        context[:,1]=1
    target=target.repeat(zero_time2,1,1).to(device).to(torch.float)
    targets=torch.concat((target_zero,target))
    input_data=torch.concat((input_signals,context),axis=1).to(torch.float)
    return input_data,targets

def generate_batch(batchsize):
    
    """

    Generate a batch of inputs of model

    offset: noisy signals' offset
    signals: noisy signal
    input_signals: input signals in channel 1 and 2
    context: context signals in channel 3 and 4

    Returns:
        input_data: batch of input data of model
        targets: batch of target outputs of model

    """
    
    target_zero=torch.zeros(size=(1,1)).repeat(T-zero_time2,batchsize,1,1).to(device).to(torch.float)
    zeros1=torch.zeros(size=(zero_time1,batchsize,input_shape,1)).to(device)
    zeros2=torch.zeros(size=(zero_time2,batchsize,input_shape,1)).to(device)
    offset=torch.tensor(np.random.choice([-0.5,0.5],(batchsize,2,1))).to(device)  
    x=torch.randn(size=(T-zero_time1-zero_time2,batchsize,input_shape,1)).to(device)+offset
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
        targets.append(label.repeat(zero_time2,1,1,1).to(device).to(torch.float))
    targets=torch.concat(targets,dim=1)
    targets=torch.concat((target_zero,targets))
    input_data=torch.concat((signal,context),axis=2).to(torch.float)
    return input_data,targets


