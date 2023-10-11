import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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

for hidden_shape in [100]:
    for P in [3]:
        for i in [0]:
            model=MDL_RNN_mante(input_shape,hidden_shape,output_shape,P,'double')
            _=model.load_state_dict(torch.load('./model/H_{}_P_{}_0.pth'.format(hidden_shape,P)))
            _=model.to(device)


input_data,targets,signals,offset=generate_inputs_offset()
output=model(input_data)
s1=plt.plot(np.arange(0,T)*2,signals[:,0].cpu(),c='red',alpha=0.3,label='signals 1')
s1=plt.plot(np.arange(0,T)*2,input_data[:,0].cpu().detach().flatten(0),c='red',alpha=1,label='signals 1')
plt.axis('off')
plt.figure()
s1=plt.plot(np.arange(0,T)*2,signals[:,1].cpu(),c='green',alpha=0.3,label='signals 1')
s1=plt.plot(np.arange(0,T)*2,input_data[:,1].cpu().detach().flatten(0),c='green',alpha=1,label='signals 1')
plt.axis('off')

plt.figure()
s1=plt.plot(np.arange(0,T)*2,output.cpu().detach().flatten(0),c='red',alpha=1,label='signals 1')
plt.axis('off')


