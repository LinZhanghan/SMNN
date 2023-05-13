import numpy as np
from train import *
from model import *
from generate_data import *

"""

Main codes for training a MDL model for contextual-dependent task

L: storage list for training loss

"""

if __name__=='__main__':
    model = MDL_RNN_mate(input_shape,hidden_shape,output_shape,P)
    model.initialize()
    model = model.to(device)
    train_losses=[]
    L=[]
    opt=optimizer(model.parameters(), lr=1e-3,betas=[0.9,0.999])
    for trial in np.arange(n):
        _=model.train()
        L.append(model.l.cpu().detach())
        input_data,targets=generate_batch(batchsize)
        for epoch in np.arange(epochs_num):
            loss=train(model,targets,input_data,opt)
            print(loss)
        train_losses.append(loss)
        print(trial,train_losses[-1])

    test(model)
