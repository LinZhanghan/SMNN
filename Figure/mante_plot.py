import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mante_utils import *
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot_input(input_data,targets,signals,offset):
    fig=plt.figure(figsize=(20,10))
    ax=fig.add_subplot(221)
    s1=plt.plot(np.arange(0,T)*2,signals[:,0].cpu(),c='deepskyblue',alpha=0.3,label='signals 1')
    lns1=plt.plot(np.arange(0,T)*2,input_data[:,0].cpu(),c='deepskyblue',alpha=1,label='stimulus 1')
    lns3=plt.plot(np.arange(0,T)*2,torch.abs(targets[:,0].cpu())*offset[0].cpu(),c='k',label='target',linewidth=3)
    plt.xlabel('Time (ms)',fontsize=30)
    #plt.legend(fontsize=25)
    ax=fig.add_subplot(222)
    s2=plt.plot(np.arange(0,T)*2,signals[:,1].cpu(),c='deeppink',alpha=0.3,label='signals 2')
    lns1=plt.plot(np.arange(0,T)*2,input_data[:,1].cpu(),c='deeppink',alpha=1,label='stimulus 2')
    
    lns3=plt.plot(np.arange(0,T)*2,torch.abs(targets[:,0].cpu())*offset[1].cpu(),c='k',label='target',linewidth=3)
    plt.xlabel('Time (ms)',fontsize=30)
    #plt.legend(fontsize=25)
    return fig,s1,s2

def test_plot(model):
    target_plot=torch.tensor([[1],[-1]]).repeat(zero_time2,1,1).to(device).to(torch.float)  
    target_zero_plot=torch.zeros(size=(input_shape-2,1)).repeat(T-zero_time2,1,1).to(device).to(torch.float)
    targets_plot=torch.concat((target_zero_plot,target_plot))
    op=[]
    on=[]
    model.eval()
    for i in range(100):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            o=model(input_data).flatten(0)
        if targets[-1]<0:
            on.append(o)
        else:
            op.append(o)
    op=torch.stack(op,0)
    on=torch.stack(on,0)
    op_mean=torch.mean(op,dim=[0]).cpu()
    op_std=torch.std(op,dim=[0]).cpu()
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(T)*2,op_mean,c='b')
    plt.fill_between(np.arange(T)*2,op_mean-op_std,op_mean+op_std,color='b',alpha=0.3)

    on_mean=torch.mean(on,dim=[0]).cpu()
    on_std=torch.std(on,dim=[0]).cpu()
    plt.plot(np.arange(T)*2,on_mean,c='r')
    plt.fill_between(np.arange(T)*2,on_mean-on_std,on_mean+on_std,color='r',alpha=0.3)

    plt.plot(np.arange(T)*2,targets_plot[:,0].cpu(),label='target')
    plt.plot(np.arange(T)*2,targets_plot[:,1].cpu(),label='target')
    plt.fill_between(np.arange(50,250)*2,-1.1,1.1,color='grey',alpha=0.3)
    plt.ylim(-1.1,1.1)
    plt.xlabel('Time (ms)',fontsize=30)
    plt.ylabel('Outputs',fontsize=30)


def plot_attractor(model,a=30,b=-60,c=30,d=-60):    
    fig1 = plt.figure(figsize=(15,8),dpi=200)
    ax1 = fig1.add_subplot(121,projection='3d')
    ax2 = fig1.add_subplot(122,projection='3d')
    cms=['Reds','Blues','Oranges','Greens']
    for i in range(100):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            _=model(input_data)
        if targets[-1]>0:
            if input_data[-1,2]>0:
                cm=cms[0]
            else:
                cm=cms[1] 
        elif targets[-1]<0:
            if input_data[-1,2]>0:
                cm=cms[2]
            else:
                cm=cms[3]
        cmap=plt.get_cmap(cm)
        rin=torch.matmul(model.pin.T,model.R).cpu().detach()
        x,y,z=torch.split(rin,1,dim=-2)
        colors=cmap(torch.arange(x.shape[0]))
        s1=ax1.scatter(x, y, z,c=torch.arange(x.shape[0]),cmap=cm,alpha=0.1,s=50)
        rout=torch.matmul(model.pout.T,model.R).cpu().detach()
        x,y,z=torch.split(rout,1,dim=-2)
        s2=ax2.scatter(x, y, z,c=torch.arange(x.shape[0]),cmap=cm,alpha=0.1,s=50) 

    labels=['left attractor','right attractor','red attractor','green attractor']
    colors=['#d62728','#ff7f0e','#1f77b4','#2ca02c']
    for i in range(4):
        ax2.scatter([], [], [],c=colors[i],marker='.',label=labels[i],s=100) 


    ax1.set_title(r'$(\xi^{in})^T r$',fontsize=30)
    ax2.set_title(r'$(\xi^{out})^T r$',fontsize=30)

    for ax in [ax1,ax2]:
        ax.set_xlabel(r'X',fontsize=30,labelpad=20)
        ax.set_ylabel(r'Y',fontsize=30,labelpad=20)
        ax.set_zlabel(r'Z',fontsize=30,labelpad=20)

    ax2.legend(loc=(-0.45,0.6),fontsize=25,markerscale=2.2)

    ax1.view_init(a,b)
    ax2.view_init(c,d)



def plot_data(input_data,targets):
    fig=plt.figure(figsize=(15,8),dpi=200)
    ax=fig.add_subplot(111)
    lns2=plt.plot(np.arange(input_data.shape[0]),input_data[:,1].cpu(),c='green',alpha=1,label='stimulus 2')
    lns1=plt.plot(np.arange(input_data.shape[0]),input_data[:,0].cpu(),c='red',alpha=1,label='stimulus 1')
    lns3=plt.plot(np.arange(input_data.shape[0]),targets.flatten(0).cpu(),c='k',label='target',linewidth=3)
    plt.xlabel('Time (ms)',fontsize=30)
    ax.set_ylabel('Stimulus',fontsize=30)
    ax=ax.twinx()
    lns4=plt.plot(np.arange(input_data.shape[0]),input_data[:,2].cpu(),'--',c='red',label='context 1',linewidth=5)
    lns5=plt.plot(np.arange(input_data.shape[0]),input_data[:,3].cpu(),'--',c='green',label='context 2',linewidth=5)

    ax.set_ylabel('Context',fontsize=30)
    lns = lns1+lns2+lns3+lns4+lns5
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, ncol=1,loc=(0.7,0.5),fontsize=25,frameon=True)

def plot_swtich_trace(model,choose,a=30,b=-60,C=30,d=-60):
    fig1 = plt.figure(figsize=(15,8),dpi=200)
    ax1 = fig1.add_subplot(121,projection='3d')
    ax2 = fig1.add_subplot(122,projection='3d')

    colors=['#d62728','#ff7f0e','#1f77b4','#2ca02c']

    for i in range(20):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            _=model(input_data)
        if targets[-1]>0:
            if input_data[-1,2]>0:
                c=colors[0]
            else:
                c=colors[2]
        elif targets[-1]<0:
            if input_data[-1,2]>0:
                c=colors[1]
            else:
                c=colors[3]

        rin=torch.matmul(model.pin.T,model.R)[-10:].cpu().detach()
        x,y,z=torch.split(rin,1,dim=-2)
        s1=ax1.scatter(x, y, z,c=c,marker='s',s=100,alpha=1)
        rout=torch.matmul(model.pout.T,model.R)[-10:].cpu().detach()
        x,y,z=torch.split(rout,1,dim=-2)
        s2=ax2.scatter(x, y, z,c=c,marker='s',s=100,alpha=1)


    labels=['left attractor','right attractor','red attractor','green attractor']

    for i in range(4):
        ax2.scatter([], [], [],c=colors[i],marker='s',label=labels[i],s=100) 


    if choose[0]=='+':
        flag1=1
    else:
        flag1=-1
    if choose[2]=='+':
        flag2=1
    else:
        flag2=-1
    if choose[1]=='1':
        flag3=2
    else:
        flag3=3
    j=0
    while(j<=0):
        #input_data,targets=generate_context_switch()
        input_data,targets=generate_random_batch_switch(1,200,1)
        if targets[250]!=flag1 or targets[-1]!=flag2 or input_data[250,:,flag3]!=1:
            continue
        _=model(input_data)
        cm='Greys'
        cmap=plt.get_cmap(cm)
        rin=torch.matmul(model.pin.T,model.R).cpu().detach()
        x,y,z=torch.split(rin,1,dim=-2)
        colors=cmap(np.linspace(0,1,x.shape[0]))
        x_s,y_s,z_s=torch.split(rin[T-1:T],1,dim=-2)
        ax1.scatter(x, y, z,c=torch.arange(x.shape[0]),cmap=cm,alpha=1,s=30)
        rout=torch.matmul(model.pout.T,model.R).cpu().detach()
        x,y,z=torch.split(rout,1,dim=-2)
        x_s,y_s,z_s=torch.split(rout[T-1:T],1,dim=-2) 
        ax2.scatter(x, y, z,c=torch.arange(x.shape[0]),cmap=cm,alpha=1,s=30)
        j+=1


    ax1.set_title(r'$(\xi^{in})^T r$',fontsize=30)
    ax2.set_title(r'$(\xi^{out})^T r$',fontsize=30)
    for ax in [ax1,ax2]:
        ax.set_xlabel(r'X',fontsize=30,labelpad=20)
        ax.set_ylabel(r'Y',fontsize=30,labelpad=20)
        ax.set_zlabel(r'Z',fontsize=30,labelpad=20)
    ax2.legend(loc=(-0.45,0.55),fontsize=25,markerscale=1)
    
    ax1.view_init(a,b)
    ax2.view_init(C,d)
    #plt.suptitle(choose,fontsize=30)
    plt.show()

def plot_projection(model,ax1,choose,a=30,b=-60):

    colors=['#d62728','#ff7f0e','#1f77b4','#2ca02c']

    for i in range(10):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            _=model(input_data)
        if targets[-1]>0:
            if input_data[-1,2]>0:
                c=colors[0]
            else:
                c=colors[2]
        elif targets[-1]<0:
            if input_data[-1,2]>0:
                c=colors[1]
            else:
                c=colors[3]

        mu=get_coff(model)
        rin=mu[-10:].cpu().detach()
        x,y,z=torch.split(rin,1,dim=-2)
        s1=ax1.scatter(x, y, z,c=c,marker='s',s=100,alpha=1)


    labels=['left attractor','right attractor','red attractor','green attractor']
    for i in range(4):
        ax1.scatter([], [], [],c=colors[i],marker='s',label=labels[i],s=100) 


    if choose[0]=='+':
        flag1=1
    else:
        flag1=-1
    if choose[2]=='+':
        flag2=1
    else:
        flag2=-1
    if choose[1]=='1':
        flag3=2
    else:
        flag3=3
    j=0
    while(j<1):
        #input_data,targets=generate_context_switch()
        input_data,targets=generate_random_batch_switch(1,200,1)
        if targets[250]!=flag1 or targets[-1]!=flag2 or input_data[250,:,flag3]!=1:
            continue
        _=model(input_data)
        cm='Greys'
        mu=get_coff(model)
        x,y,z=torch.split(mu.detach(),1,dim=-2)
        x_s,y_s,z_s=torch.split(mu.detach()[T-1:T],1,dim=-2)
        s1=ax1.scatter(x, y, z,c=torch.arange(x.shape[0])*2,cmap=cm,alpha=1,s=30)

        j+=1


    #ax1.set_title('$\mu$',fontsize=30)
    ax1.set_xlabel(r'$X$',fontsize=30,labelpad=20)
    ax1.set_ylabel(r'$Y$',fontsize=30,labelpad=25)
    ax1.set_zlabel(r'$Z$',fontsize=30,labelpad=20)
    ax1.view_init(a,b)
    ax1.legend(loc=(0.8,0.75),fontsize=25)
    return input_data,targets