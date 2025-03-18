import torch
import torch.nn as nn
import numpy as np
from snntorch import spikegen
from torch.nn import init
from model_settings import *


# Define Network
class MDL_RNN_mate(nn.Module):

    """

    MDL Spiking model for contextual-dependent task

    """

    def __init__(self,input_shape,hidden_shape,output_shape,P,filter='double',mdl=1,tref=2,replace=0):

        """

        num_inputs: input size
        num_hidden: number of hidden neurons
        num_outputs: output size
        filter: synaptic filter
        P: mode size
        Win: input weight
        Wout: output weight
        pin: input mode
        pout: output mode
        l: connectivity importance
        mdl: flag for using mdl
        tref: refractory period
        replace: flag for replacing step function
        Wr : connectivity matrix for standard fully connection

        """    

        super().__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.P=P
        self.filter=filter
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.mdl=mdl
        self.tref=tref
        self.replace=replace
        if mdl:    
            self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
            self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
            self.l=torch.nn.Parameter(torch.randn(size=[self.P]),requires_grad=True)
        else:
            self.Wr=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.hidden_shape)),requires_grad=True)

        
    def forward(self, inputs):

        """

        Forward dynamics for model

        mem: membrane potential
        r: storage variable for filtered firing rates
        s: storage variable for spikes
        U: storage list for membrane potential
        R: storage list for filtered firing rates
        S: storage list s
        I: external current
        Wr: connectivity matrix
        
        Returns
            matmul(W_out,R)

        """

        # Construct connectivity matrix
        if self.mdl:
            Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        else:
            Wr=self.Wr.to(device)

        # Initialize variables
        U=[]
        S=[]
        R=[]
        if len(inputs.shape)>3:
            batch_size=inputs.shape[1]
        else:
            batch_size=1
        time_steps=inputs.shape[0]
        if self.filter=='double': 
            lr=torch.exp(-dt/tau_r).to(device)
            h = torch.zeros(size=(self.hidden_shape,1)).to(device)
        tref=self.tref
        tlast=torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)-1
        I = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        s = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/tau_m).to(device)
        ld=torch.exp(-dt/tau_d).to(device)
        r = torch.zeros(size=(self.hidden_shape,1)).to(device)

        #Dynamics
        for i in range(time_steps):
            I=torch.matmul(self.Win,inputs[i])+torch.matmul(Wr,r)
            mem=(dt*i>(tlast+tref))*(lm*mem+(1-lm)*I)*(1-s)
            if self.filter=='single':
                r=ld*r+dt/tau_d*s  
            elif self.filter=='double':
                h=lr*h+s
                r=ld*r+(1-ld)*h  
            if self.replace:
                s=spike_grad_approx(mem-vthr)
            else:
                s=spike_grad(mem-vthr)
            tlast=tlast+(dt*i-tlast)*s
            U.append(mem)
            S.append(s)
            R.append(r)
        S=torch.stack(S,0)
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        self.U=U
        self.inputs=inputs
        self.S=S
        self.R=R
        
        #Generate output
        rout=torch.matmul(self.Wout,R)
        return rout
        
    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.input_shape))
            elif 'pin' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.P))
            elif 'pout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.P))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=np.sqrt(self.P)/np.sqrt(self.hidden_shape))
            elif 'Wout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))
            elif 'Wr' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))


class rate_RNN_mante(nn.Module):

    """

    MDL rate model for contextual-dependent task

    """

    def __init__(self,input_shape,hidden_shape,output_shape,P,mdl=1):
        
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
        mdl: flag for using mdl
        Wr : connectivity matrix for standard fully connection

        """            

        super().__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.l=torch.nn.Parameter(torch.zeros(size=[self.P])+1/self.hidden_shape,requires_grad=False)
        self.mdl=mdl
        if mdl:    
            self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
            self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
            self.l=torch.nn.Parameter(torch.randn(size=[self.P]),requires_grad=True)
        else:
            self.Wr=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.hidden_shape)),requires_grad=True)

    def forward(self, inputs):

        """

        Forward dynamics for model

        mem: membrane potential
        r: storage variable for filtered firing rates
        U: storage list for membrane potential
        R: storage list for filtered firing rates
        I: external current
        Wr: connectivity matrix
        
        Returns
            matmul(W_out,R)

        """
        # Construct connectivity matrix    
        if self.mdl:
            Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        else:
            Wr=self.Wr.to(device)

        # Initialize variables
        R=[]
        U=[]
        if len(inputs.shape)>3:
            batch_size=inputs.shape[1]
        else:
            batch_size=1
        time_steps=inputs.shape[0]
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/tau_m).to(device)

        #Dynamics
        for i in range(time_steps):
            I=torch.matmul(self.Win,inputs[i])+torch.matmul(Wr,torch.tanh(mem))
            mem = lm*mem+(1-lm)*I
            U.append(mem)
            R.append(torch.tanh(mem))
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        self.U=U
        self.R=R
        self.inputs=inputs

        #Generate output
        rout=torch.matmul(self.Wout,R)
        return rout

    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.input_shape))
            elif 'pin' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.P))
            elif 'pout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.P))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=np.sqrt(self.P)/np.sqrt(self.hidden_shape))
            elif 'Wout' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))
            elif 'Wr' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))
