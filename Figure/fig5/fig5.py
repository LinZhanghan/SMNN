import matplotlib.pyplot as plt
import numpy as np
from Figure.mnist_utils import *
from Figure.mnist_model import *
import torch
N_range=[30,50,100,200]
fig=plt.figure(figsize=(10,8),dpi=200)
ax1=fig.add_subplot(111)
name='train_loss'
c=['steelblue','deepskyblue','skyblue','lightskyblue']
c=['lightcoral','indianred','firebrick','darkred']
c=['tan','peru','darkorange','chocolate']
for hidden_shape in [30,100,200]:
    R=[]
    L=[]
    j=np.argwhere(np.array(N_range)==hidden_shape)[0,0]
    for P in [200]:
        for i in range(3):
            model=MDL_RNN_mnist(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load('Figure/mante/model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            a=torch.load('Figure/mante/{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            if a[-1]>=70: continue
            r=rank(model)
            r,indces=torch.sort(r)
            r=r.flip(0)
            L.append(torch.abs(model.l)[indces.flip(0)].cpu().detach())
            R.append(r)
    R=torch.stack(R,0)
    L=torch.stack(L,0)
    mean=torch.mean(L,0)
    std=torch.std(L,0)
    ax1.plot(torch.arange(mean.shape[0]),mean,c=c[j],linewidth=3)            
    ax1.fill_between(torch.arange(std.shape[0]),mean-std,mean+std,color=c[j],alpha=0.3)
    mean=torch.mean(R,0)
    std=torch.std(R,0)
    #ax1.plot(torch.arange(mean.shape[0]),mean,c='deepskyblue')            
    #ax1.fill_between(torch.arange(std.shape[0]),mean-std,mean+std,color='skyblue',alpha=0.3)
    #ax1.plot([],[],c='deepskyblue',label=r'$\tau$')
    ax1.plot([],[],c=c[j],label='N={}'.format(hidden_shape),linewidth=3)
    ax1.set_xlabel('Rank',fontsize=30)
    ax1.set_ylabel('$|\Sigma|$',fontsize=30)
    
    ax1.set_ylim(-0.1,3)

    left, bottom, width, height = 0.55, 0.45, 0.3, 0.2
    ax = fig.add_axes([left, bottom, width, height])
    ax.plot(torch.arange(mean.shape[0]),mean,c=c[j],linewidth=3)            
    ax.fill_between(torch.arange(std.shape[0]),mean-std,mean+std,color=c[j],alpha=0.3)
    ax.plot([],[],c='deepskyblue')
    ax.vlines(4,-0.1,3,linestyle='--',linewidth=2)
    ax.vlines(180,-0.1,3,linestyle='--',linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Rank',fontsize=25)
    ax.set_ylabel(r'$\tau$',fontsize=25)
    ax.xaxis.set_ticks([1,10,100])
    
ax1.legend(fontsize=25,ncol=2,columnspacing=0.4)