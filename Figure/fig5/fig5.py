import matplotlib.pyplot as plt
import numpy as np
from Figure.mnist_utils import *
from Figure.mnist_model import *
import torch
N_range=[30,50,100,200]
fig=plt.figure(figsize=(10,8),dpi=200)
ax1=fig.add_subplot(111)
name='train_loss'
fig=plt.figure(figsize=(10,8),dpi=200)
ax1=fig.add_subplot(111)
name='train_loss'
c=['#3cb44b','#4363d8']
N_range=[100,200]
for hidden_shape in N_range:
    R=[]
    L=[]
    j=np.argwhere(np.array(N_range)==hidden_shape)[0,0]
    P=30
    for i in range(3):
        model=MDL_RNN_mnist(input_shape,hidden_shape,output_shape,P,'double')
        _=model.load_state_dict(torch.load('./model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
        _=model.to(device)
        a=torch.load('./{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
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
    ax1.plot(torch.arange(mean.shape[0])+1,mean,c=c[j],linewidth=3)            
    ax1.fill_between(torch.arange(std.shape[0])+1,mean-std,mean+std,color=c[j],alpha=0.3)
    mean=torch.mean(R,0)
    std=torch.std(R,0)
    ax1.plot([],[],c=c[j],label='N={}'.format(hidden_shape),linewidth=3)
    ax1.set_xlabel('Rank',fontsize=30)
    ax1.set_ylabel('$|\lambda_{\mu}|$',fontsize=30)
    ax1.set_ylim(-0.1,3)
    left, bottom, width, height = 0.55, 0.55, 0.3, 0.2
    ax = fig.add_axes([left, bottom, width, height])
    ax.plot(torch.arange(mean.shape[0])+1,mean,c=c[j],linewidth=3)            
    ax.fill_between(torch.arange(std.shape[0])+1,mean-std,mean+std,color=c[j],alpha=0.3)
    ax.plot([],[],c='deepskyblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(top=4.5,bottom=0.4)
    ax.set_yticks([0.5,1,2,3])
    ax.set_yticklabels([0.5,1,2,3])
    ax.set_xticks([1,10])
    ax.set_xlabel('Rank',fontsize=25)
    ax.set_ylabel(r'$\tau$',fontsize=25)
    if hidden_shape==100:
        d_range=[[0,9],[9,19]]
    else:
        d_range=[[0,10],[10,18]]
    for d1,d2 in d_range:
        x=torch.arange(mean.shape[0])[d1:d2].numpy()+1
        y=mean[d1:d2].numpy()
        x=np.log(x)
        y=np.log(y)
        from scipy.optimize import curve_fit
        def func(x, a,b):
            return a*x+b
        p0=[-0.1,0]
        popt, pcov = curve_fit(func, x, y,p0=p0)
        yvals = func(x,*popt)
        #plt.plot(x,y)
        ax.plot(np.exp(x), np.exp(yvals), '-.',c='grey',linewidth=3)
        if d1==0:
            ax.text(np.exp(x[int((d2-d1)/6)]-0.5),np.exp(yvals[int((d2-d1)/6)])*2-3,r'slope$\approx$'+'{:.2f}'.format(popt[0]),fontsize=12)
        else:
            ax.text(np.exp(x[int((d2-d1)/6)]-0.5),np.exp(yvals[int((d2-d1)/6)])*2.15-2.8,r'slope$\approx$'+'{:.2f}'.format(popt[0]),fontsize=12)
        #R2
        ybar=np.sum(y)/len(y)
        ssreg=np.sum((yvals-ybar)**2)
        sstot=np.sum((y-ybar)**2)
        a=ssreg/sstot
        print(a)

ax.vlines(9,0.4,4.5,linestyle='--',linewidth=2,color=c[0])
ax.vlines(19,0.4,4.5,linestyle='--',linewidth=2,color=c[0])
ax.vlines(10,0.4,4.5,linestyle='--',linewidth=2,color=c[1])
ax.vlines(18,0.4,4.5,linestyle='--',linewidth=2,color=c[1])
ax.text(7.8,0.45,'$9$',fontsize=10,color=c[0])
ax.text(19,0.45,'$19$',fontsize=10,color=c[0])
ax.text(10,0.45,'$10$',fontsize=10,color=c[1])
ax.text(14,0.45,'$18$',fontsize=10,color=c[1])

ax1.legend(fontsize=25,ncol=2,columnspacing=0.4)