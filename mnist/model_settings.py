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
time_steps: time steps
batchsize: batch size for training
device: cpu or gpu device for running
spike_grad: surrogate delta function
dt: time internal
train_loader: data loader for training
test_loader: data loader for testing
"""

spike_grad = surrogate.fast_sigmoid()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])
data_path='./data/mnist'
time_steps = 100
input_shape=784
num_classes=10
dt=0.2
batch_size = 256
n=10
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=500, shuffle=True)

P=3
hidden_shape=100

