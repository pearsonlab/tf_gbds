# tf-gbds
TensorFlow rewrite of [goal-based dynamical system model](https://github.com/pearsonlab/gbds)

Implementation of

> S Iqbal, J Pearson (2017). [Learning to Draw Dynamic Agent Goals with Generative Adversarial Networks](https://arxiv.org/abs/1702.07319v1)

Code for approximate time series posterior written by Evan Archer. Algorithm described in

>  E Archer, IM Park, L Buesing, J Cunningham, L Paninski (2015). [Black box variational inference for state space models](http://arxiv.org/abs/1511.07367)

## How to Preprocess Your Data
Our model reads in data from an experiment as an hdf5 file. Only one variable is required for the model to train: `Trajectories`, which our code expects to be a matrix with the following shape: (nTrials, nTimepoints, nDimensions). So, in a variant of Penalty Kick with 9000 total trials, each trial consisting of 80 timepoints, and each timepoint consisting of 3 dimensions (goalie y-position, ball x-position, ball y-position--see Iqbal & Pearson, 2017), then the `trajectories` variable will have shape (9000, 80, 3). 

- Optional Variables that can be inputted into model
1. `Conditions` (such as subID, type of opponent, of perhaps drug condition (i.e. saline v. muscimol)), in the shape of (nTrials, conditions). So in an example with 50 subjects, each with a unique one-hot-encoded subID, conditions will have shape (9000, 50). 
1. `Result` (was the trial a win or loss) with shape (nTrials, 1)
1. `Control`, representing joystick inputs,  with shape (nTrials, nTimepoints, nDimensions).