import torch
import torch.nn as nn
from snntorch import surrogate
from snntorch import spikegen
spike_grad = surrogate.fast_sigmoid()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Define Network
T=50
class MDL_RNN_mnist(nn.Module):
    def __init__(self,input_shape, hidden_shape,output_shape,P,filter='single'):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.filter=filter
        self.P=P
        self.Win=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.input_shape)),requires_grad=True)
        self.Wout=torch.nn.Parameter(torch.randn(size=(self.output_shape,self.hidden_shape)),requires_grad=True)
        self.pin=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)#输入模式
        self.pout=torch.nn.Parameter(torch.randn(size=(self.hidden_shape,self.P)),requires_grad=True)#输出模式
        self.l=torch.nn.Parameter(torch.zeros(size=[self.P]),requires_grad=True)#连接重要性

    def forward(self, x):
        U=[]
        S=[]
        R=[]
        batch_size=x.shape[0]
        dt=2e-3
        time_steps=T
        vthr=torch.tensor(1.0)
        taus=torch.tensor(10e-3)
        taum=torch.tensor(20e-3)
        tau_d=torch.tensor(30e-3)
        spk_in = spikegen.rate(x.flatten(1),time_steps).reshape(time_steps,-1,784,1)
        if self.filter=='double': 
            tau_r=torch.tensor(2e-3)
            lr=torch.exp(-dt/tau_r).to(device)
            h = torch.zeros(size=(self.hidden_shape,1)).to(device)
        tref=5*dt
        tlast=torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)-1
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        I = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        s = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        ls=torch.exp(-dt/taus).to(device)
        lm=torch.exp(-dt/taum).to(device)
        ld=torch.exp(-dt/tau_d).to(device)
        r = torch.zeros(size=(self.hidden_shape,1)).to(device)
        for i in range(time_steps):
            I=ls*I+torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,r)
            mem=(dt*i>(tlast+tref))*(lm*mem+(1-lm)*I)*(1-s)
            if self.filter=='single':
                r=torch.exp(-dt/tau_d)*r+dt/tau_d*s  
            elif self.filter=='double':
                h=lr*h+dt/tau_d/tau_r*s
                r=ld*r+dt*h  
            s=spike_grad(mem-vthr)
            tlast=tlast+(dt*i-tlast)*s
            U.append(mem)
            S.append(s)
            R.append(r)
        S=torch.stack(S,0)
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        rm,_=torch.max(R,0)
        rout=torch.matmul(self.Wout,rm)
        y = torch.softmax(rout,1)
        self.U=U
        self.spk=spk_in
        self.S=S
        self.R=R
        self.rm=rm
        return y


class rate_RNN_mnist(nn.Module):
    def __init__(self,input_shape, hidden_shape,output_shape,P):
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

    def forward(self, x):
        R=[]
        U=[]
        batch_size=x.shape[0]
        dt=2e-3
        time_steps=T
        Wr=torch.matmul(self.l*self.pin,self.pout.T).to(device)
        taum=torch.tensor(20e-3)
        spk_in = spikegen.rate(x.flatten(1),time_steps).reshape(time_steps,-1,784,1)
        mem = torch.zeros(size=(batch_size,self.hidden_shape,1)).to(device)
        lm=torch.exp(-dt/taum).to(device)
        for i in range(time_steps):
            I=torch.matmul(self.Win,spk_in[i])+torch.matmul(Wr,torch.tanh(mem))
            mem = lm*mem+(1-lm)*I
            U.append(mem)
            R.append(torch.tanh(mem))
        U=torch.stack(U,0)
        R=torch.stack(R,0)
        Rm,_=torch.max(R,0)
        Rout=torch.matmul(self.Wout,Rm)
        y = torch.softmax(Rout,1)
        self.R=R
        self.U=U
        self.spk=spk_in
        return y