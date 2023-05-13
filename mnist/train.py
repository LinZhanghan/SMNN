import torch
import torch.nn as nn
import torch.nn.functional as F
from model_settings import *

def train(model,targets, inputs,opt):

    """

    Methods to train model wish optimizer

    opt: optimizer
    targets: target outputs
    inputs: input data

    Returns
        loss.item(): loss function

    """
    model.to(device)
    output = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    loss=criterion(output, targets)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=30, norm_type=2)
    opt.step()
    return loss.item()

def test(model):

    """

    Methods to train model wish optimizer

    test_losses: test loss
    correct: correct number of test
    
    Returns:
        correct / len(test_loader.dataset): accuracy rate

    """
    
    test_losses=[]
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.reshape((-1,28,28)).to(device)
            targets=torch.nn.functional.one_hot(target).float().to(device)
            target=target.to(device)
            output= model(data)
            test_loss += F.cross_entropy(output, targets,reduction='sum').item()
            pred=output.data.max(1)[1]
            correct+=pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


