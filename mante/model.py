import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from model_settings import *


class MDL_RNN_mante(nn.Module):

    """

    MDL RNN model for contextual-dependent task

    """

    def __init__(self,input_shape,hidden_shape,output_shape,P):

        """

        num_inputs: input size
        num_hidden: number of hidden neurons
        num_outputs: output size
        filter: synaptic filter
        P: mode size
        Win: input matrix
        Wout: output matrix
        pin: input mode
        pout: output mode
        l: connectivity importance

        """    

        super().__init__()
        self.num_inputs = input_shape
        self.num_hidden = hidden_shape
        self.num_outputs = output_shape
        self.filter=filter
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.l=torch.nn.Parameter(torch.zeros(size=[self.P]),requires_grad=True)

        
    def forward(self, x):

        """

        Forward dynamics for model

        x: input image
        Wr: connectivity matrix
        U: membrane potential of hidden neuron
        S: storage list for spike train
        R: storage list for firing rate
        rm: the maximum value of firing rate
        
        Returns
            matmul(W_out,R)

        """

        U=[]
        S=[]
        R=[]
        if len(x.shape)>3:
            batch_size=x.shape[1]
        else:
            batch_size=1

        vthr=torch.tensor(1.0)
        taus=torch.tensor(10e-3)
        taum=torch.tensor(20e-3)
        tau_d=torch.tensor(30e-3)
        ls=torch.exp(-dt/taus).to(device)
        lm=torch.exp(-dt/taum).to(device)
        ld=torch.exp(-dt/tau_d).to(device)
        tlast=torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)-1
        tref=5*dt
        if self.filter=='double': 
            tau_r=torch.tensor(2e-3)
            lr=torch.exp(-dt/tau_r).to(device)
            h = torch.zeros(size=(self.hidden_shape,1)).to(device)
        
        spk_in=x
        
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        
        I = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        s = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        r = torch.zeros(size=(self.hidden_shape,1)).to(device)
        
        for i in range(time_steps):
            I=ls*I+(torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,r))
            mem=(dt*i>(tlast+tref))*(lm*mem+(1-lm)*I)*(1-s)
            if self.filter=='single':
                r=ld*r+dt/tau_d*s  
            elif self.filter=='double':
                h=lr*h+s
                r=ld*r+(1-ld)*h  
            s=spike_grad(mem-vthr)
            tlast=tlast+(dt*i-tlast)*s
            U.append(mem)
            S.append(s)
            R.append(r)
            
        S=torch.stack(S,0)
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        rout=torch.matmul(self.Wout,R)
        y=rout
        self.U=U
        self.spk=spk_in
        self.S=S
        self.R=R
        return y

    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in model.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(input_shape))
            elif 'pin' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(P))
            elif 'pout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(P))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(hidden_shape))
            elif 'Wout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(output_shape))


class rate_RNN_mante(nn.Module):

    """

    rate RNN model for contextual-dependent task

    """

    def __init__(self,input_shape, hidden_shape,output_shape,P):

        """

        num_inputs: input size
        num_hidden: number of hidden neurons
        num_outputs: output size
        P: mode size
        Win: input matrix
        Wout: output matrix
        pin: input mode
        pout: output mode
        l: connectivity importance

        """    

        super().__init__()
        self.num_inputs = input_shape
        self.num_hidden = hidden_shape
        self.num_outputs = output_shape
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.l=torch.nn.Parameter(torch.zeros(size=[self.P]),requires_grad=True)

    def forward(self, x):

        """

        Forward dynamics for model

        x: input image
        Wr: connectivity matrix
        U: membrane potential of hidden neuron
        R: storage list for firing rate
        rm: the maximum value of firing rate
        
        Returns 
            matmul(Wout,R)

        """
        U=[]
        R=[]
        if len(x.shape)>3:
            batch_size=x.shape[1]
        else:
            batch_size=1

        taum=torch.tensor(20e-3)
        lm=torch.exp(-dt/taum).to(device)
        
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        
        spk_in = x
        
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        
        for i in range(time_steps):
            I=torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,torch.tanh(mem))
            mem = lm*mem+(1-lm)*I
            U.append(mem)
            R.append(torch.tanh(mem))
            
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        y=torch.matmul(self.Wout,R)
        self.U=U
        self.R=R
        self.spk=spk_in
        return y

    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in model.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(input_shape))
            elif 'pin' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(P))
            elif 'pout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(P))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(hidden_shape))
            elif 'Wout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(output_shape))


