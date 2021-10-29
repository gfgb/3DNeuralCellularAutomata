# 3D Neural Cellular Automata
Project for the course of **Deep Learning and Applied Artificial Intelligence 2020/2021**\
Master's Degree in Computer Science\
Sapienza University of Rome
## Report
## Description

This project is an extension to the 3-dimensional domain of the work\
*Growing Neural Cellular Automata* by A. Mordvintsev, E. Randazzo, E. Niklasson, M. Levin\
https://distill.pub/2020/growing-ca/ \
\
Part of the code is adapted from https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata



## Results

### Growing
Targets
<p float="left">
  <img src="/models/bunny_growing_e5000_h300_target.png" width="300" />
  <img src="/models/camel_growing_e5000_h300_target.png" width="300" /> 
</p>

https://user-images.githubusercontent.com/28317156/139420469-3c009481-6ee9-4710-b3cc-03c5eae46e1b.mp4

https://user-images.githubusercontent.com/28317156/139420516-ea83191e-9ba6-4556-b18a-c4fef715252f.mp4


### Persisting

Targets
<p float="left">
  <img src="/models/bunny_regenerating_e5000_h300_target.png" width="300" />
  <img src="/models/camel_regenerating_e5000_h300_target.png" width="300" />
</p>

https://user-images.githubusercontent.com/28317156/139425673-b1284d6f-2d06-4b40-b32b-7d833abbc8db.mp4

https://user-images.githubusercontent.com/28317156/139420726-991c0e3a-908f-4219-979f-a810237d90c2.mp4


### Regenerating







## How to run

**Training**
```
usage: main.py [-h] [--name NAME] [--file FILE] [-exp EXPERIMENT_TYPE]
               [--lr LR] [--lr-decay LR_DECAY] [--betas BETAS BETAS]
               [--epochs EPOCHS] [--batch-sz BATCH_SZ] [--pool-sz POOL_SZ]
               [--steps STEPS STEPS] [--hidden-sz HIDDEN_SZ]
               [--n-channels N_CHANNELS] [--cell-fire-rate CELL_FIRE_RATE]
               [--padding PADDING]

optional arguments:
  -h, --help            show this help message and exit

global:
  --name NAME           Name of the experiment (used for saving files)
  --file FILE           Name of the json data file to use
  -exp EXPERIMENT_TYPE, --experiment-type EXPERIMENT_TYPE
                        Experiment type (growing, persisting, regenerating)

training:
  --lr LR               Training learning rate
  --lr-decay LR_DECAY   Training learning rate decay
  --betas BETAS BETAS   Coefficients for Adam
  --epochs EPOCHS       Number of training epochs
  --batch-sz BATCH_SZ   Batch size
  --pool-sz POOL_SZ     Pool size
  --steps STEPS STEPS   Min and max steps for each training epoch

network:
  --hidden-sz HIDDEN_SZ
                        Size of the hidden layer of the NN
  --n-channels N_CHANNELS
                        Number of auxiliary channels for the NN
  --cell-fire-rate CELL_FIRE_RATE
                        Cell fire rate

data:
  --padding PADDING     Padding applied to the voxelized data
```
**Rendering**
```
```
