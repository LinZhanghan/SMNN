import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from model_settings import *


# Define Network
class MDL_RNN_mate(nn.Module):

    """

    MDL RNN model for contextual-dependent task

    """

    def __init__(self,input_shape,hidden_shape,output_shape,P):

        """

        num_inputs: input size
        num_hidden: number of hidden neurons
        num_outputs: output size
        P: mode size
        Win: input weight
        Wout: output weight
        pin: input mode
        pout: output mode
        l: connectivity importance
        lif: LIF neurons
        sigmoid_tau_d: tau_d Gaussian variable for sigmoid

        """    

        super().__init__()
        self.num_inputs = input_shape
        self.num_hidden = hidden_shape
        self.num_outputs = output_shape
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.num_hidden,self.num_inputs+2)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.num_outputs,self.num_hidden)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.num_hidden,self.P)),requires_grad=True)
        self.pout=torch.nn.Parameter(torch.randn(size=(self.num_hidden,self.P)),requires_grad=True)
        self.l=torch.nn.Parameter(torch.randn(size=[self.P]),requires_grad=True)
        self.sigmoid_tau_d=torch.nn.Parameter(torch.randn(size=(self.num_hidden,1)),requires_grad=True)
        self.matmul=torch.matmul
        self.stack=torch.stack
        self.exp=torch.exp
        self.tanh=torch.tanh
        self.sigmoid=torch.sigmoid
        self.lif=snn.Leaky(beta=0.9,threshold=0.5,spike_grad=spike_grad,learn_beta=False)
        self.diag=torch.diag

        
    def forward(self, x):

        """

        Forward dynamics for model

        mem: membrane potential
        spk: spike train
        r: filtered firing rates
        s: storage variable for filtered firing rates
        U: storage list for membrane potential
        Spk: storage list for spike train
        R: storage list for filtered firing rates
        S: storage list s
        dt: time internal
        tau_r: time constant tau_r
        tau_d: time constant tau_d
        I: external current
        Wr: connectivity matrix
        
        Returns
            matmul(W_out,R)

        """

        U=[]
        Spk=[]
        R=[]
        S=[]
        dt=5e-3
        time_steps=x.shape[0]
        r = torch.zeros(size=(self.num_hidden,1)).to(device)
        s = torch.zeros(size=(self.num_hidden,1)).to(device)
        mem=self.initmem()
        tau_min=2e-2
        tau_step=3e-2
        tau_d=self.sigmoid(self.sigmoid_tau_d)*tau_step+tau_min
        tau_r=torch.tensor([2e-3]).to(device)
        # Construct connectivity matrix
        Wr=self.matmul(self.l*self.pin,self.pout.T).to(device)

        #RNN dynamics
        for i in range(time_steps):    
            I=self.matmul(self.Win,x[i])+self.matmul(Wr,r)+torch.randn(size=(self.num_hidden,1)).to(device)/10
            spk ,mem = self.lif(I,mem)
            s=s*self.exp(-dt/tau_r)+dt/tau_d/tau_r*spk
            r=self.exp(-dt/tau_d)*r+dt*s
            R.append(r)
            U.append(mem)
            Spk.append(spk)
            S.append(s)

        Spk=self.stack(Spk,0)
        R=self.stack(R,0)
        S=self.stack(S,0)
        U=self.stack(U,0)
        self.spk=Spk
        self.r=R
        self.s=S
        self.U=U
        
        #Generate output
        y=self.matmul(self.Wout,R)
        return y

    def initmem(self):

        """

        Method to initialize the membrane potential
        
        Returns
            lif.init_leaky(): initial membrane potential
            
        """

        return self.lif.init_leaky()
    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1)
            elif 'p' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.num_hidden))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.num_hidden))
            elif 'Wout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.num_hidden))



