import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import snntorch.spikeplot as splt
params = {
         'figure.figsize': ((15,8)),
         'legend.fontsize': 'x-large',
         'xtick.labelsize':20,
         'ytick.labelsize':20,
         'axes.labelsize': 'xx-large',
         'axes.spines.top': True,
         'axes.spines.right': True,
         'axes.titlesize': 'xx-large',}
pylab.rcParams.update(params)
plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 200 #分辨率

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from Figure.mnist_utils import *
from Figure.mnist_model import *
P_range=[1,2,3,4,5,10,30,50,100,200]
N_range=[30,50,100,200]
rate_path='Figure/mnist/rate'


# Figure 2a
fig = plt.figure(figsize=(8,5),dpi=200)
ax = fig.add_subplot()
name='train_loss'
for hidden_shape in N_range:
    results=[]
    for P in P_range:
        time=[]
        for i in np.arange(3):
            a=torch.load('Figure/mnist/{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            acc=torch.load('Figure/mnist/{}/H_{}_P_{}_{}.pth'.format('acc',hidden_shape,P,i))
            if acc[-1]<0.89: continue
            a=torch.where(a<1.7)[0][0]
            time.append(a)
        time=torch.stack(time,0).float().numpy()
        results.append(time)
    
    mean=np.array([np.mean(x) for x in results])
    std=np.array([np.std(x) for x in results])

    err = np.zeros([2,std.shape[0]])
    err[0,:] = std
    err[1,:] = std
    _=ax.errorbar(P_range,mean,yerr=err[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$N$={}'.format(hidden_shape))
plt.xscale('log')
plt.legend(fontsize=25,ncol=2)
plt.ylabel('Training trials',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)
plt.ylim(0,600)

#Figure 2b
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
P_range=[1,2,3,10,50,100]
name='train_loss'
for P in P_range:
    results=[]
    for N in N_range[2:3]:
        loss=[]
        for i in range(3):
            a=torch.load('Figure/mnist/{}/H_{}_P_{}_{}.pth'.format(name,N,P,i))
            if a[-1]<1.7:
                loss.append(a[:500:20])
            else:
                continue
        loss=torch.stack(loss,0).float()
        results.append(loss)
    results=torch.tensor(results[0])
    err = np.zeros([2,results.shape[1]])
    mean=results[0]
    std=results[1]

    _=ax.errorbar(np.arange(mean.shape[0])*20,mean,yerr=err[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$P$={}'.format(P))
plt.legend(fontsize=25,ncol=2,columnspacing=0.4)
plt.ylabel('Cross Entropy',fontsize=30)
plt.xlabel('Training batch',fontsize=30)

#Figure 2c
filter='double'
acc_mean=[]
acc_std=[]
for hidden_shape in N_range:
    test_acc=[]
    test_std=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            model=MDL_RNN_mnist(input_shape,hidden_shape,output_shape,P,filter)
            _=model.load_state_dict(torch.load('Figure/mnist/saved_model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            accuracy=torch.load('Figure/mnist/{}/H_{}_P_{}_{}.pth'.format('acc',hidden_shape,P,i))
            if accuracy[-1]<0.89: continue
            for n in range(2):
                acc.append(test(model,3000).cpu())
        test_acc.append(np.mean(acc)*100)
        test_std.append(np.std(acc)*100)
    acc_mean.append(test_acc)
    acc_std.append(test_std)

acc_mean_rate=[]
acc_std_rate=[]
for hidden_shape in N_range:
    test_acc=[]
    test_std=[]
    for P in P_range:
        acc=[]
        for i in range(3):
            model=rate_RNN_mnist(input_shape,hidden_shape,output_shape,P)
            _=model.load_state_dict(torch.load(rate_path+'/model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            accuracy=torch.load(rate_path+'/{}/H_{}_P_{}_{}.pth'.format('acc',hidden_shape,P,i))
            if accuracy[-1]<0.89: continue
            for n in range(2):
                acc.append(test(model,3000).cpu())
        test_acc.append(np.mean(acc)*100)
        test_std.append(np.std(acc)*100)
    acc_mean_rate.append(test_acc)
    acc_std_rate.append(test_std)

fig = plt.figure(figsize=(8,5),dpi=200)
i=0
n=30
for test_acc,test_std in zip(acc_mean,acc_std):
    if N_range[i]!=n: 
        i+=1
        continue
    err = np.zeros([2,len(test_acc)])
    err[0,:] = test_std
    err[1,:] = test_std
    plt.errorbar(P_range,test_acc[::-1],yerr=err[:,::-1],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='spike')
    i+=1
i=0
for test_acc,test_std in zip(acc_mean_rate,acc_std_rate):
    if N_range[i]!=n: 
        i+=1
        continue
    err = np.zeros([2,len(test_acc)])
    err[0,:] = test_std
    err[1,:] = test_std
    plt.errorbar(P_range,test_acc[::-1],yerr=err[:,::-1],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='rate')
    i+=1
plt.legend(loc='upper center',fontsize=25,ncol=2)
plt.ylabel('Test acc. (%)',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)
plt.xscale('log')
plt.ylim(91.5,100)

#Figure 2d
loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
data,_,_=generate_input(loader)
with torch.no_grad():
    o=model(data)
T=50
U=model.U.cpu().detach()
r=model.R.cpu().detach()
r=r.reshape(r.shape[0],-1)
U=U.reshape(U.shape[0],-1)
index=torch.argmax(r,dim=0)
index=index[U[-1,:]>0]
U=U[:,U[-1,:]>0]
Um=U[index,:].diag()

a=0
b=3
U=U[:,a:b]

U=torch.where(U>1,1,U)
fig = plt.figure(facecolor="w", figsize=(12,8),dpi=200)
ax = fig.add_subplot(111)
plt.plot(np.arange(T)*2,U)
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('U',fontsize=30)
marker=['s','v','^']
for i in range(b-a):
    plt.scatter(index[a:b][i]*2,Um[a:b][i],marker=marker[i],s=200,label='argmax(r)',alpha=1)
plt.ylim(torch.min(U)-0.1,torch.max(U)+0.1)
plt.hlines(0,0,T*2,colors = "k",linewidth=5, linestyles = "dashed")
plt.text(0,-0.15,'$U_{res}$',fontsize=20)
plt.hlines(1,0,T*2,colors = "r",linewidth=5, linestyles = "dashed")
plt.text(0,1.1,'$U_{thr}$',fontsize=20)
plt.ylim(top=1.5)
plt.legend(fontsize=25)


#Figure 2e
from matplotlib import transforms
from matplotlib.gridspec import GridSpec
model=MDL_RNN_mnist(input_shape,100,output_shape,10)
_=model.load_state_dict(torch.load('Figure/mnist/saved_model/H_{}_P_{}_0.pth'.format(100,10)))
_=model.to(device)
loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
input_data,targets,_=generate_input(loader)
with torch.no_grad():
    o=model(input_data).flatten(0)
spk=model.S.flatten(1).cpu()
fig = plt.figure(facecolor="w", figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax = fig.add_subplot(gs[0:2,0:2])
splt.raster(spk, ax, s=10, c="chocolate")
plt.gca().xaxis.set_visible(False)
plt.ylabel('Neuron',fontsize=30)
plt.ylim(-5,105)
#plt.xlim(-10,510)
ax = fig.add_subplot(gs[2,0:2])
s1=plt.plot(torch.arange(spk.shape[0]),spk.sum(1)/100/2e-3,c='tan',linewidth=2,label='firing rate \n     (HZ)')
plt.gca().set_xticklabels(['0','0','20','40','60','80','100'])
plt.xlabel('Time (ms)',fontsize=30)

ax = fig.add_subplot(gs[0:2,2:])
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(-90)
s2=plt.plot(torch.arange(spk.shape[1]),spk.sum(0)/0.1,label='firing times',linewidth=2,c='tan',transform=rot + base)
plt.gca().yaxis.set_visible(False)
plt.ylim(5,-105)
fig.subplots_adjust(wspace=0,hspace=0)
lns = s1
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, ncol=1,loc=(0.02,-0.4),fontsize=25,frameon=True)