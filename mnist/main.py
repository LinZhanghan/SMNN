import torch
import numpy as np
from train import *
from model import *

"""

Main codes for training a MDL model for contextual-dependent task

train_losses: storage list for training loss
acc: storage list for accuracy rate

"""

if __name__=="__main__":
   
    model = MDL_RNN_mnist(input_shape,hidden_shape,num_classes,P)
    model.initialize()
    model = model.to(device)
    train_losses=[]
    acc=[]
    print(model.parameters())

    opt = optimizer(model.parameters(), lr=0.001,betas=[0.9,0.999])
   

    for epoch in np.arange(10*n+1):
        _=model.train()
        loss=0
        for i in range(5):
            examples = enumerate(train_loader)
            batch_idx, (example_data, example_targets) = next(examples)
            example_data=example_data.reshape((batch_size,28,28)).to(device)
            example_targets=torch.nn.functional.one_hot(example_targets).float().to(device)
            loss+=train(model,example_targets,example_data,opt)
        train_losses.append(loss/10)
        if epoch%10==0: 
            print("epoch {} train loss: ".format(epoch),train_losses[-1])
            acc.append(test(model).cpu())



