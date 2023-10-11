# MDL RNNs

## SMNN model

This repository provides the code for the framework presented in [this paper](to be decided):

Given the current revolution of AI techniques (e.g., Chat GPT), a new learning framework named mode decomposition learning (MDL) is introduced, calling for a rethinking of conventional weight-based deep learning through the lens of cheap and interpretable mode-based learning. The combination of MDL and Spiking Neural Network (SNN) is more energy-efficient and efficient, and can achieve complex contextual integration and image recognition tasks. MDL projects the hign-dimensional network performance into low-dimensional to visualize and analysis, displaying a attractor phenomenon and a striking piecewise power-law behavior.

## Requirements

The code for constructing and training  is implemented in Python (tested in Python 3.8.5). The code also requires torch and snntorch

- torch 2.0.0+cu117
- snntorch 0.6.2
- numpy 1.19.2


## Usage
The code for training models for mnist task is located in `mnist/`, while the code for tranning models for contextual-dependent task is in `mante/`. Code for plotting Figure is in `Figure/`.

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
  - `dt`: time interval.
  - `train_loader`: data loader for training.
  - `test_loader`: data loader for testing.
  - `optmizer`: optimizer.

You can train the model with,

```
python minist/main.py 
```

 The training losses will and test accuracy be storaged in `train_losses` and `acc`.
 
 
### Contextual-dependent task

The settings file (`mante/model_settings.py`) contains the following input arguments:
  - `P`: mode size.
  - `hidden_shape`: number of neurons.
  - `input_shape`: numbers of input signals.
  - `output_shape`: output size.
  - `n`: trials of training.
  - `T`: time steps.
  - `batchsize`: batch size for training.
  - `device`: cpu or gpu device for running.
  - `spike_grad`: surrogate delta function.
  - `epochs_num`: epochs for each trials.
  - `zero_time1`: zero time beform stimulus.
  - `zero_time2`: zero time after stimulus.
  - `target_zero`: target output of zero.
  - `dt`: time interval.
  - `zero1`: zero input beform stimulus.
  - `zero2`: zero input beform stimulus.
  - `optmizer`: optimizer.

You can train the model with,

```
python mante/main.py 
```

 The training losses will be storaged in `L` and a test plot over 100 trials will be drawn.

## Citation
This code is the product of work carried out by the group of [PMI lab, Sun Yat-sen University](https://www.labxing.com/hphuang2018). If the code helps, consider giving us a shout-out in your publications.

## Contact
If you have any question, please contact me via [linchh26@mail2.sysu.edu.cn](linchh26@mail2.sysu.edu.cn).
