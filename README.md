# tf-gbds
TensorFlow+Edward rewrite version of [goal-based dynamical system model](https://github.com/pearsonlab/gbds)

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
export PYTHONPATH=$PYTHONPATH:/path/to/your/directory/tf_gbds/
```
where "path/to/your/directory" is replaced with the path to the tf_gbds repository. This allows the nested directories to access modules from their parent directory.

## Train an tf_gbds model

Once you prepare your dataset with the correct format, you can train the model!
```
# Run tf_gbds
$ python run_model.py --model_type='VI_KLqp' (Type of model to build {VI_KLqp, HMM}) \
--model_dir='/path/you/save/your/model' (Directory where the model is saved) \
--max_sessions=10 (Maximum number of sessions to load) \
--session_type='recording' (Type of data session) \
--session_index_dir='/directory/you/save/session_index_file' (Directory of session index file) \
--data_dir='/directory/you/load/your/data' (Directory of data file) \
--synthetic_data=False (Is the model trained on synthetic data?) \
--save_posterior=True (Will posterior samples be retrieved after training?) \
--load_saved_model=False (Is the model restored from a checkpoint?)
--saved_model_dir='/directory/you/save/restored_model' (Directory where the model to be restored is saved) \
--device_type='cpu' (The device where the model is trained {CPU, GPU}) \

--p1_dim=1 (Number of data dimensions corresponding to player 1) \
--p2_dim=2 (Number of data dimensions corresponding to player 2) \

--rec_lag=10 (Number of previous timepoints to include as input to recognition model) \
--rec_nlayers=3 (Number of layers in recognition model neural networks) \
--rec_hidden_dim=25 (Number of hidden units in each (dense) layer of recognition model neural networks) \

--gen_nlayers=3 (Number of layers in generative model neural networks) \
--gen_hidden_dim=25 (Number of hidden units in each (dense) layer of generative model neural networks) \
--K=8 (Number of sub-strategies (components of GMM)) \

--add_accel=False (Should acceleration be added to states?) \
--clip=True (Is the control signal censored?) \
--clip_range=1. (The range beyond which control signals are censored) \
--clip_tol=1e-5 (The tolerance of signal censoring) \
--eta=1e-6 (The scale of control signal noise) \
--eps_init=1e-5 (Initial value of control signal variance) \
--eps_trainable=False (Is epsilon trainable?) \
--eps_penalty=None (Penalty on control signal variance) \
--sigma_init=1e-5 (Initial value of goal state variance) \
--sigma_trainable=False (Is sigma trainable?) \
--sigma_penalty=None (Penalty on goal state variance) \
--goal_bound=1.0 (Goal state boundaries) \
--goal_bound_penalty=1e10 (Penalty for goal states escaping boundaries) \

--seed=1234 (Random seed) \
--train_ratio=0.85 (The proportion of data used for training) \
--optimizer='Adam' (Training optimizer) \
--learning_rate=1e-3 (Initial learning rate) \
--num_epochs=500 (Number of iterations through the full training set) \
--batch_size=128 (Size of mini-batch) \
--num_samples=1 (Number of posterior samples for calculating stochastic gradients) \
--num_posterior_samples=30 (Number of samples drawn from posterior distributions) \
--max_ckpt_to_keep=5 (maximum number of checkpoint to save) \
--frequency_val_loss=5 (frequency of saving validate loss) \
```
## Visualize a training model

To visualize training variables and loss curves while training tf_gbds model, run the following command on your model directory:

```sh
tensorboard --logdir=/directory/where/you/saved/your/graph
```
