import torch
import torch.nn as nn
from snntorch import spikegen
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from model_settings import *

# Define Network
class MDL_RNN_mnist(nn.Module):

    """

    MDL spiking model for mnist task

    """

    def __init__(self,input_shape, hidden_shape,output_shape,P,filter='double',mdl=1,tref=2,replace=0):

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

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.filter=filter
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=False)
        self.tref=tref
        self.mdl=mdl
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

        inputs: input image
        mem: membrane potential for coding
        r: storage variable for filtered firing rates
        s: storage variable for spikes
        spk_in: input spike train
        Wr: connectivity matrix
        U: membrane potential of hidden neuron
        S: storage list for spike train
        R: storage list for firing rate
        rm: the maximum value of firing rate
        
        Returns 
            softmax(Wout * Um + Bias)

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
        batch_size=inputs.shape[0]
        spk_in = spikegen.rate(inputs.flatten(1),T).reshape(T,-1,784,1)
        if self.filter=='double': 
            tau_r=torch.tensor(2)
            lr=torch.exp(-dt/tau_r).to(device)
            h = torch.zeros(size=(self.hidden_shape,1)).to(device)
        tref=self.tref
        tlast=torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)-1
        I = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        s = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/taum).to(device)
        ld=torch.exp(-dt/tau_d).to(device)
        r = torch.zeros(size=(self.hidden_shape,1)).to(device)

        #Dynamics
        for i in range(T):
            I=torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,r)
            mem=(dt*i>(tlast+tref))*(lm*mem+(1-lm)*I)*(1-s)
            if self.filter=='single':
                r=ld*r+dt/tau_d*s  
            elif self.filter=='double':
                h=lr*h+dt/tau_d/tau_r*s
                r=ld*r+dt*h  
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
        rm,_=torch.max(R,0)
        self.U=U
        self.inputs=inputs
        self.spk=spk_in
        self.S=S
        self.R=R
        self.rm=rm

        #Generate output
        rout=torch.matmul(self.Wout,rm)
        output = torch.softmax(rout,1)
        return output
    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))
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

class rate_RNN_mnist(nn.Module):
    
    """

    MDL rate model for mnist task

    """

    def __init__(self,input_shape, hidden_shape,output_shape,P,mdl=1):

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
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)#输入模式
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)#输出模式
        self.l=torch.nn.Parameter(torch.zeros(size=[self.P])+1/self.hidden_shape,requires_grad=False)#连接重要性
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

        inputs: input image
        mem: membrane potential for coding
        r: storage variable for filtered firing rates
        spk_in: input spike train
        Wr: connectivity matrix
        U: membrane potential of hidden neuron
        R: storage list for firing rate
        rm: the maximum value of firing rate
        
        Returns 
            softmax(Wout * Um + Bias)

        """

        # Construct connectivity matrix
        if self.mdl:
            Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        else:
            Wr=self.Wr.to(device)

        # Initialize variables
        R=[]
        U=[]
        batch_size=inputs.shape[0]
        spk_in = spikegen.rate(inputs.flatten(1),T).reshape(T,-1,784,1)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/tau_m).to(device)

        #Dynamics
        for i in range(T):
            I=torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,torch.tanh(mem))
            mem = lm*mem+(1-lm)*I
            U.append(mem)
            R.append(torch.tanh(mem))
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        rm,_=torch.max(R,0)
        self.R=R
        self.U=U
        self.inputs=inputs
        self.spk=spk_in

        #Generate output
        rout=torch.matmul(self.Wout,rm)
        output = torch.softmax(rout,1)
        return y
    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'Win' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(self.hidden_shape))
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
