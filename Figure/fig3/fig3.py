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

from Figure.mante_utils import *
from Figure.mante_model import *
from Figure.mante_plot import *

P_range=[1,2,3,4,5,10,30,50,100,200]
N_range=[30,50,100,200]

from sklearn.metrics import mean_squared_error
def test_r2():
    R2=[]
    for i in range(5):
        input_data,targets=generate_inputs()
        with torch.no_grad():
            o=model(input_data).flatten(0).cpu()
        r=r2(targets.flatten(0).cpu(),o)
        R2.append(r)
    return  np.array(R2)
def r2(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)



# Figure 3a
R=[]
name='train_loss'
for hidden_shape in N_range:
    R2=[]
    for P in P_range:
        r=[]
        for i in range(3):
            model=model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load('Figure/mante/model/H_{}_P_{}_{}.pth'.format(hidden_shape,P,i)))
            _=model.to(device)
            a=torch.load('Figure/mante/{}/H_{}_P_{}_{}.pth'.format(name,hidden_shape,P,i))
            if a[-1]>=80:
                continue
            r.append(test_r2())
        R2.append(np.concatenate(r))
    R2=np.array(R2)
    R.append(R2)
mode_R=R
i=0
fig, ax = plt.subplots(1, 1, figsize=(8, 5),dpi=200)
for R2 in mode_R:
    R2_mean=np.array([np.mean(x) for x in R2])
    R2_std=np.array([np.std(x) for x in R2])
    yerr = np.zeros([2,len(R2_mean)])
    yerr[0,:] = R2_std
    yerr[1,:] = R2_std
    ax.errorbar(P_range,R2_mean,yerr=yerr[:,:],ecolor='k',elinewidth=0.5,marker='.',\
        mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$N$={}'.format(N_range[i]))
    i+=1
plt.legend(loc=1,ncol=2,fontsize=25,columnspacing=0.4)
plt.xscale('log')
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Mode Size ($P$)',fontsize=30)

#Figure 3b
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot()
R=[]
name='train_loss'
for P in P_range:
    results=[]
    for N in N_range[2:3]:
        loss=[]
        for i in range(3):
            a=torch.load('Figure/mante/{}/H_{}_P_{}_{}.pth'.format(name,N,P,i))
            if a[-1]<100:
                loss.append(a[::50])
            else:
                continue
        loss=torch.stack(loss,0).float()
        results.append(loss)
    results=torch.tensor(results[0])
    err = np.zeros([2,results.shape[1]])
    mean=results[0]
    std=results[1]

    _=ax.errorbar(np.arange(mean.shape[0])*50,mean,yerr=err[:,:],ecolor='k',elinewidth=0.5,marker='.',\
    mec='k',mew=1,ms=30,alpha=1,capsize=5,capthick=3,linestyle="--",label='$P$={}'.format(P))

plt.legend(loc=1,ncol=2,fontsize=25,columnspacing=0.4)
plt.ylabel('MSE',fontsize=30)
plt.xlabel('Training batch',fontsize=30)

#Figure 3c
for hidden_shape in [100]:
    for P in [3]:
        for i in [0]:
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load('Figure/mante/model/H_{}_P_{}_double.pth'.format(hidden_shape,P)))
            _=model.to(device)
test_plot(model)

#Figure 3d
U=model.U.cpu().detach()
r=model.R.cpu().detach()
r=r.reshape(r.shape[0],-1)
U=U.reshape(U.shape[0],-1)
index=torch.argmax(r,dim=0)
Um=U[index,:].diag()
U=U[:,U[-1,:]>0]
U=U[:,:3]
U=torch.where(U>1,1,U)
fig = plt.figure(facecolor="w", figsize=(12,8),dpi=200)
ax = fig.add_subplot(111)
plt.plot(np.arange(T)*2,U)
plt.xlabel('Time (ms)',fontsize=30)
plt.ylabel('U',fontsize=30)
plt.fill_between(np.arange(50,250)*2,torch.min(U)-3,5,color='grey',alpha=0.3)
#plt.scatter(index[[10:13]]*2,Um[[10:13]],c='k',marker='^')
plt.ylim(torch.min(U)-0.1,2)
plt.hlines(0,0,T*2,colors = "k",linewidth=5, linestyles = "dashed")
plt.text(0,-0.45,'$U_{res}$',fontsize=20)
plt.hlines(1,0,T*2,colors = "r",linewidth=5, linestyles = "dashed")
plt.text(0,1.15,'$U_{thr}$',fontsize=20)


#Figure 3e
from matplotlib import transforms
from matplotlib.gridspec import GridSpec
input_data,targets=generate_inputs()
with torch.no_grad():
    o=model(input_data).flatten(0)
spk=model.S.flatten(1).cpu()
fig = plt.figure(facecolor="w", figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax = fig.add_subplot(gs[0:2,0:2])
splt.raster(spk, ax, s=10, c="chocolate")
plt.gca().xaxis.set_visible(False)
plt.ylabel('Neuron',fontsize=30)
plt.fill_between(np.arange(50,250),-10,110,color='grey',alpha=0.3)
plt.ylim(-5,105)
plt.xlim(-10,510)
ax = fig.add_subplot(gs[2,0:2])
s1=plt.plot(torch.arange(spk.shape[0]),spk.sum(1)/100/2e-3,c='tan',linewidth=2,label='firing rate \n     (HZ)')
plt.gca().set_xticklabels(['0','0','200','400','600','800','1000'])
plt.xlabel('Time (ms)',fontsize=30)
plt.fill_between(np.arange(50,250),-10,100,color='grey',alpha=0.3)
plt.ylim(-5,80)
plt.xlim(-10,510)
ax = fig.add_subplot(gs[0:2,2:])
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(-90)
s2=plt.plot(torch.arange(spk.shape[1]),spk.sum(0),label='firing times',linewidth=2,c='tan',transform=rot + base)
plt.gca().yaxis.set_visible(False)
plt.ylim(5,-105)
fig.subplots_adjust(wspace=0,hspace=0)
lns = s1
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, ncol=1,loc=(0.02,-0.4),fontsize=25,frameon=True)