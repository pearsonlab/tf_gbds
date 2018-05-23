# tf-gbds
TensorFlow+Edward implementation of [goal-based dynamical system model](https://github.com/pearsonlab/gbds)

Implementation of

> S Iqbal, J Pearson (2017). [Learning to Draw Dynamic Agent Goals with Generative Adversarial Networks](https://arxiv.org/abs/1702.07319v1)

Code for approximate time series posterior written by Evan Archer. Algorithm described in

>  E Archer, IM Park, L Buesing, J Cunningham, L Paninski (2015). [Black box variational inference for state space models](http://arxiv.org/abs/1511.07367)


## Contents
- Python Modules/Scripts
1. `run_model.py` Train tf_gbds model. Run `python run_model.py --help` to see [options](#train-an-tf_gbds-model).
1. `GenerativeModel.py` The customized Edward Random Variables which generate players' goal and control signal for each time point based on the trajectory of each trial.
1. `RecognitionModel.py` The customized Edward Random Variables which implement
the posterior goal and control signal usiing smoothing linear dynamical system.
1. `utils.py` The utility functions needed for run_model.
1. `lib/` The directory containing libraries needed for matrix computation.

## Prerequisites
The code is written in Python 3.6.1. You will also need:
* **TensorFlow-gpu** version 1.6.0 ([install](https://www.tensorflow.org/install/)) -
* **Edward** version 1.3.5 ([install](http://edwardlib.org/getting-started))
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)

## Before you start
Run the following:
```sh
export PYTHONPATH=$PYTHONPATH:/path/to/your/directory/tf_gbds/
```
where "path/to/your/directory" is replaced with the path to the tf_gbds repository. This allows the nested directories to access modules from their parent directory.

## Visualize a training model

To visualize training variables and loss curves while training tf_gbds model in tensorboard, run the following command on your model directory:

```sh
tensorboard --logdir=/directory/where/you/saved/your/graph
```
