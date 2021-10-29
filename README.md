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

The target shape is achieved but, as the model keeps updating the shape, the result degenerates.

https://user-images.githubusercontent.com/28317156/139420469-3c009481-6ee9-4710-b3cc-03c5eae46e1b.mp4

https://user-images.githubusercontent.com/28317156/139420516-ea83191e-9ba6-4556-b18a-c4fef715252f.mp4


### Persisting

Targets
<p float="left">
  <img src="/models/bunny_regenerating_e5000_h300_target.png" width="300" />
  <img src="/models/camel_regenerating_e5000_h300_target.png" width="300" />
</p>

By implementing the sampling pool strategy, the model is able to impose the target shape as an "attractor" for incorrect states, so that degenerations are redirected towards the target.

https://user-images.githubusercontent.com/28317156/139425673-b1284d6f-2d06-4b40-b32b-7d833abbc8db.mp4

https://user-images.githubusercontent.com/28317156/139420726-991c0e3a-908f-4219-979f-a810237d90c2.mp4


### Regenerating

Targets
<p float="left">
  <img src="/models/bunny_regenerating_e5000_h300_target.png" width="300" />
  <img src="/models/camel_regenerating_e5000_h300_target.png" width="300" />
</p>

By training the model with damaged samples, the model learns to regenerate missing areas of the shape.
Here, at frames 50 and 110 a spherical area of the volume is "removed".

https://user-images.githubusercontent.com/28317156/139432257-874a8470-ed42-49a9-b43b-39862e75e9c3.mp4

https://user-images.githubusercontent.com/28317156/139432274-3d796978-9b59-4ac9-bce7-4cdb2dc3e1ac.mp4

### Regenerating with a trainable convolutional layer
Targets
<p float="left">
  <img src="/models/camel_conv_regenerating_e5000_h300_target.png" width="300" />
</p>

https://user-images.githubusercontent.com/28317156/139439854-549efa28-e3cd-484d-a558-fb79b7e770a5.mp4

## How to run

**Training**
```
usage: main.py [-h] [--name NAME] [--file FILE] [-exp EXPERIMENT_TYPE] [--lr LR] [--lr-decay LR_DECAY]
               [--betas BETAS BETAS] [--epochs EPOCHS] [--batch-sz BATCH_SZ] [--pool-sz POOL_SZ] [--steps STEPS STEPS]
               [--trainable-conv TRAINABLE_CONV] [--hidden-sz HIDDEN_SZ] [--n-channels N_CHANNELS]
               [--cell-fire-rate CELL_FIRE_RATE] [--padding PADDING]

optional arguments:
  -h, --help            show this help message and exit

global:
  --name NAME           Name of the experiment
  --file FILE           Name of the data file to use
  -exp EXPERIMENT_TYPE, --experiment-type EXPERIMENT_TYPE
                        Experiment type

training:
  --lr LR               Training learning rate
  --lr-decay LR_DECAY   Training learning rate decay
  --betas BETAS BETAS   Coefficients for Adam
  --epochs EPOCHS       Number of training epochs
  --batch-sz BATCH_SZ   Batch size
  --pool-sz POOL_SZ     Pool size
  --steps STEPS STEPS   Min and max steps for each training epoch

network:
  --trainable-conv TRAINABLE_CONV
                        Use trainable convolutional layer
  --hidden-sz HIDDEN_SZ
                        Size of the hidden layer of the NN
  --n-channels N_CHANNELS
                        Number of auxiliary channels for the NN
  --cell-fire-rate CELL_FIRE_RATE
                        Random dropout of cells from updating to simulate asynchronicity

data:
  --padding PADDING     Padding applied to the voxelized data
```
Example
```
python main.py --name camel --trainable-conv 0 --file camel.json --experiment-type regenerating --epochs 5000 --hidden-sz 300
```
**Rendering**
```
usage: render.py [-h] [--experiment-name EXPERIMENT_NAME] [--steps STEPS] [--fps FPS] [--azim AZIM]
                 [--rotation-speed ROTATION_SPEED] [-dmg DAMAGE DAMAGE DAMAGE DAMAGE DAMAGE]

optional arguments:
  -h, --help            show this help message and exit

global:
  --experiment-name EXPERIMENT_NAME
                        Name of the experiment and name of the output video
  --steps STEPS         Number of updates to perform
  --fps FPS             Framerate
  --azim AZIM           Starting position of the camera wrt the center
  --rotation-speed ROTATION_SPEED
                        Rotation of the camera
  -dmg DAMAGE DAMAGE DAMAGE DAMAGE DAMAGE, --damage DAMAGE DAMAGE DAMAGE DAMAGE DAMAGE
                        A tuple (x, y, z, r, f) where x, y and z are the coordinates of the center of a sphere; r is
                        its radius; f is the time (as number of updates) when you want the damage to be applied The
                        sphere is used to damage the volume (i.e. set areas of the volume to zero)
```
Example
```
python render.py --experiment-name camel_regenerating_e5000_h300 --steps 200 --fps 15 --azim 0 --rotation-speed 1 -dmg 10 2 10 7 50 -dmg 10 16 15 6 110
```
