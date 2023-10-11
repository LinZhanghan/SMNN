import torch
import torch.nn as nn
from model_settings import *

# Define Network
class MDL_RNN_mnist(nn.Module):

    """

    MDL RNN model for mnist task

    """

    def __init__(self,input_shape, hidden_shape,output_shape,P,,filter='single'):

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
        mem: membrane potential for coding
        spk: spike train
        spk_rec: storage list for spike train
        cur: flatten image as input current for LIF neurons
        Wr: connectivity matrix
        U: membrane potential of hidden neuron
        Um: the maximum value of membrane potential
        Returns 
            softmax(Wout Um + Bias)

        """

        U=[]
        spk_rec=[]
        Wr=self.matmul(self.l*self.pin,self.pout.T).to(device)
        cur=self.fc1(x.flatten(1))
        mem=self.initmem()
        
        for i in range(time_steps):
          spk ,mem = self.lif(cur,mem)
          spk_rec.append(spk)
        spk=self.stack(spk_rec,2)
        #Convolution operations
        conv=self.conv(spk,self.filter,groups=self.num_hidden,padding='same')
        U=self.matmul(Wr,conv)
        Um,_=self.max(U,2)
        y = self.fc2(Um)
        y = self.softmax(y)
        self.U=U
        self.spk=spk
        return y

    def initmem(self):

        """

        Method to initialize the membrane potential
        
        Returns
            lif.init_leaky(): initial membrane potential

        """
        return self.lif.init_leaky()

    def linar_filter(self,t,V0=1,tau=0.4,taus=0.1,ti=1e-6):

        """

        Method to construct the linear kernel
        
        Returns
            linear kenel

        """
        return V0*(torch.exp(-(t-ti)/tau)-torch.exp(-(t-ti)/taus))
    
    def initialize(self):

        """

        Method to initialize model parameters

        """
        
        for name, param in self.named_parameters():
            if 'bias' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(hidden_shape))
            elif 'beta' in name:
                _=init.constant_(param, val=0.9)
            elif 'p' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(hidden_shape))
            elif 'l' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(hidden_shape))
            elif 'fc2' in name:
                _=init.normal_(param, mean=0, std=1/np.sqrt(num_classes))
            else:
                _=init.normal_(param, mean=0, std=1/np.sqrt(input_shape))
