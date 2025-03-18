import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate

"""

Model settings for mnist task

transform: transform method for mnist images
data_path: data path for mnist dataset
P: mode size
hidden_shape: number of neurons
input_shape: numbers of input signals
output_shape: output size
n: trials of training
T: time steps
batchsize: batch size for training
device: cpu or gpu device for running
spike_grad: surrogate delta function
spike_grad_approx : smooth transfer function
dt: time step
train_loader: data loader for training
test_loader: data loader for testing
optimizer: optimizer
vthr : firing threshold
tau_m : membrane time constant
tau_d : synaptic decay time constant
tau_r : synaptic rise time constant

"""

spike_grad = surrogate.fast_sigmoid()
def spike_grad_approx(x,beta=20):
    return (1+torch.tanh(beta*x))/2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])
data_path='./data/mnist'
T = 100
input_shape=784
num_classes=10
dt=0.2
batch_size = 300
n=10
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=500, shuffle=True)

P=3
hidden_shape=100
optimizer = torch.optim.Adam
vthr=torch.tensor(1.0)
tau_m=torch.tensor(20)
tau_d=torch.tensor(30)
