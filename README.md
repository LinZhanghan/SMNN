# MDL RNNs

## Overview

This repository provides the code for the framework presented in [this paper](https://www.pnas.org/content/116/45/22811):

to be decided

Preprint available [here](to be decided).

## Requirements

The code for constructing and training  is implemented in Python (tested in Python 3.8.5). The code also requires torch and snntorch

- torch 2.0.0+cu117
- snntorch 0.6.2
- numpy 1.19.2


## Usage
The code for training models for mnist task is located in `mnist/`, while the code for tranning models for contextual-dependent task is in `mante/`.

### Mnist task

The settings file (`mnist/model_settings.py`) contains the following input arguments:
  - `transform`: transform method for mnist images.
  - `data_path`: data path for mnist dataset.
  - `P`: mode size.
  - `hidden_shape`: number of neurons.
  - `input_shape`: numbers of input signals.
  - `output_shape`: output size.
  - `n`: trials of training.
  - `time_steps`: time steps.
  - `batchsize`: batch size for training.
  - `device`: cpu or gpu device for running.
  - `spike_grad`: surrogate delta function.
  - `dt`: time internal.
  - `train_loader`: data loader for training.
  - `test_loader`: data loader for testing.

You can train the model with. The training losses will be storaged in `L`.
```
python main.py 
```


### Mapping and Constructing LIF RNN
Trained rate RNNs are used to construct LIF RNNs. The mapping and LIF simulations are performed in MATLAB.
Given a trained rate model, the first step is to perform the grid search to determine the optimal scaling factor (lambda). This is done by `lambdad_grid_search.m`. Once the optimal scaling factor is determined, a LIF RNN can be constructed using the function `LIF_network_fnc.m`. All the required functions/scripts are located in `spiking/`.

An example script for evaluating a Go-NoGo LIF network (`eval_go_nogo.m`) is also included. The script constructs a LIF RNN trained to perform the Go-NoGo task and plots network responses. The script can be modified to evaluate models trained to perform other tasks.

## Citation
If you use this repo for your research, please cite our work:

```
@article{Kim_2019,
    Author = {Kim, Robert and Li, Yinghao and Sejnowski, Terrence J.},
    Doi = {10.1073/pnas.1905926116},
    Journal = {Proceedings of the National Academy of Sciences},
    Number = {45},
    Pages = {22811--22820},
    Publisher = {National Academy of Sciences},
    Title = {Simple framework for constructing functional spiking recurrent neural networks},
    Volume = {116},
    Year = {2019}}
```
