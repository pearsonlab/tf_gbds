# tf-gbds
TensorFlow rewrite version of [goal-based dynamical system model](https://github.com/pearsonlab/gbds)

Implementation of

> S Iqbal, J Pearson (2017). [Learning to Draw Dynamic Agent Goals with Generative Adversarial Networks](https://arxiv.org/abs/1702.07319v1)

Code for approximate time series posterior written by Evan Archer. Algorithm described in

>  E Archer, IM Park, L Buesing, J Cunningham, L Paninski (2015). [Black box variational inference for state space models](http://arxiv.org/abs/1511.07367)


## Contents
- Python Modules/Scripts
1.`GenerativeModel.py` The customized Edward Random Variables which generate players' goal and control signal for each time point based on the trajectory of each trial.
1. `RecognitionModel.py` The customized Edward Random Variables which implement
the posterior goal and control signal from generative model and smoothing linear dynamical system.
1. `run_model.py` Train tf_gbds model. Run `python run_model.py --help` to see options.
1. `utils.py` The utility functions needed for run_model.

## How to Preprocess Your Data
Our model reads in data from an experiment as an hdf5 file. Only one variable is required for the model to train: `Trajectories`, which our code expects to be a matrix with the following shape: (nTrials, nTimepoints, nDimensions). So, in a variant of Penalty Kick with 9000 total trials, each trial consisting of 80 timepoints, and each timepoint consisting of 3 dimensions (goalie y-position, ball x-position, ball y-position--see Iqbal & Pearson, 2017), then the `trajectories` variable will have shape (9000, 80, 3). 

- Optional Variables that can be inputted into model
1. `Conditions` (such as subID, type of opponent, of perhaps drug condition (i.e. saline v. muscimol)), in the shape of (nTrials, conditions). So in an example with 50 subjects, each with a unique one-hot-encoded subID, conditions will have shape (9000, 50). 
1. `Result` (was the trial a win or loss) with shape (nTrials, 1)
1. `Control`, representing joystick inputs,  with shape (nTrials, nTimepoints, nDimensions).

## Prerequisites
The code is written in Python 3.6.1. You will also need:
* **TensorFlow-gpu** version 1.4.0 ([install](https://www.tensorflow.org/install/)) -
* **Edward** version 1.3.4 ([install](http://edwardlib.org/getting-started))
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)
* **h5py** ([install](https://pypi.python.org/pypi/h5py))

## Before you start
Run the following:
```sh
export PYTHONPATH=$PYTHONPATH:/<b>path/to/your/directory</b>/tf_gbds/
```

## where "path/to/your/directory" is replaced with the path to the tf_gbds repository. This allows the nested directories to access modules from their parent directory.

## Train an tf_gbds model

Once you prepare your dataset with the correct format, you can train the model!
```sh
# Run tf_gbds
$ python run_model.py --model_dir='/path/you/save/your/model' (Directory where the model is saved) \
--max_sessions=10 (Maximum number of sessions to load) \
--session_type='recording' (Type of data session) \
--session_index_dir='/directory/you/save/session_index_file' (Directory of session index file) \
--to be added
```
## Visualize a training model

To visualize training variables and loss curves while training tf_gbds model, run the following command on your model directory:

```sh
tensorboard --logdir=/directory/where/you/saved/your/graph
```