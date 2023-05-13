import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from generate_data import *

def train(model,targets, inputs,opt):

    """

    Methods to train model wish optimizer

    opt: optimizer
    targets: target outputs
    inputs: input data

    Returns
        loss.item(): loss function

    """

    model.to(device)
    outputs = model(inputs)
    criterion = torch.nn.MSELoss(reduction='sum')
    criterion.to(device)
    loss=torch.sqrt(criterion(outputs, targets))
    opt.zero_grad()
    loss.backward()    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    opt.step()
    return loss.item()

def test(model):

    """

    Methods to test model

    op: output of positive offset input
    on: output of negative offset input
    T: time steps
    targets_plot: target outputs of both offsets
    
    """
    
    target_plot=torch.tensor([[1],[-1]]).repeat(zero_time2,1,1).to(device).to(torch.float)
    target_zero_plot=torch.zeros(size=(input_shape,1)).repeat(T-zero_time2,1,1).to(device).to(torch.float)
    targets_plot=torch.concat((target_zero_plot,target_plot))
    op=[]
    on=[]
    model.eval()
    for i in range(100):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            o=model(input_data)[:,0]
        if targets[-1]<0:
            on.append(o)
        else:
            op.append(o)
    op=model.stack(op,0)
    on=model.stack(on,0)

    op_mean=torch.mean(op,dim=[0])[:,0].cpu()
    op_std=torch.std(op,dim=[0])[:,0].cpu()

    plt.plot(range(T),op_mean,c='b')
    plt.fill_between(range(T),op_mean-op_std,op_mean+op_std,color='b',alpha=0.3)

    on_mean=torch.mean(on,dim=[0])[:,0].cpu()
    on_std=torch.std(on,dim=[0])[:,0].cpu()
    plt.plot(range(T),on_mean,c='r')
    plt.fill_between(range(T),on_mean-on_std,on_mean+on_std,color='r',alpha=0.3)

    plt.plot(range(T),targets_plot[:,0].cpu(),label='target')
    plt.plot(range(T),targets_plot[:,1].cpu(),label='target')
    plt.title('hidden_shape={},P={}'.format(hidden_shape,P))
    plt.xlabel('time steps')
    plt.ylabel('outputs')



