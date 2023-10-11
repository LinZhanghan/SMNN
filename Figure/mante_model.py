import torch
import torch.nn as nn
from snntorch import surrogate
spike_grad = surrogate.fast_sigmoid()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Define Network
class MDL_RNN_mante(nn.Module):
    def __init__(self,input_shape,hidden_shape,output_shape,P,filter='single'):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.P=P
        self.filter=filter
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)
        self.l=torch.nn.Parameter(torch.randn(size=[self.P]),requires_grad=True)

    def forward(self, x):
        U=[]
        S=[]
        R=[]
        if len(x.shape)>3:
            batch_size=x.shape[1]
        else:
            batch_size=1
        dt=2e-3
        time_steps=x.shape[0]
        vthr=torch.tensor(1.0)
        taus=torch.tensor(10e-3)
        taum=torch.tensor(20e-3)
        tau_d=torch.tensor(30e-3)
        if self.filter=='double': 
            tau_r=torch.tensor(2e-3)
            lr=torch.exp(-dt/tau_r).to(device)
            h = torch.zeros(size=(self.hidden_shape,1)).to(device)
        tref=5*dt
        tlast=torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)-1
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        spk_in=x
        I = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        s = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        ls=torch.exp(-dt/taus).to(device)
        lm=torch.exp(-dt/taum).to(device)
        ld=torch.exp(-dt/tau_d).to(device)
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


class rate_RNN_mante(nn.Module):
    def __init__(self,input_shape,hidden_shape,output_shape,P):
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

    def forward(self, x):
        R=[]
        U=[]
        if len(x.shape)>3:
            batch_size=x.shape[1]
        else:
            batch_size=1
        dt=2e-3
        time_steps=x.shape[0]
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        taum=torch.tensor(20e-3)
        spk_in = x
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/taum).to(device)
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


class rate_RNN_mante_(nn.Module):
    def __init__(self,input_shape,hidden_shape,output_shape,P):
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

    def forward(self, x):
        R=[]
        U=[]
        if len(x.shape)>3:
            batch_size=x.shape[1]
        else:
            batch_size=1
        dt=2e-3
        time_steps=x.shape[0]
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        taum=torch.tensor(20e-3)
        spk_in = x
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/taum).to(device)
        for i in range(time_steps):
            I=torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,torch.tanh(mem))
            mem = lm*mem+(1-lm)*I
            U.append(mem)
            R.append(mem)
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        y=torch.matmul(self.Wout,R)
        self.U=U
        self.R=R
        self.spk=spk_in
        return y