# tf-gbds
TensorFlow + Edward implementation of [goal-based dynamical system model](https://github.com/pearsonlab/gbds).

Relevant papers:
> SN Iqbal, L Yin, CB Drucker, Q Kuang, J Gariépy, ML Platt, JM Pearson (2018). Latent goal models for dynamic strategic interaction (PLOS Computational Biology, accepted)

> S Iqbal, J Pearson (2017). [Learning to Draw Dynamic Agent Goals with Generative Adversarial Networks](https://arxiv.org/abs/1702.07319v2)

Code for approximate time series posterior is written by Evan Archer. Algorithm is described in

> E Archer, IM Park, L Buesing, J Cunningham, L Paninski (2015). [Black box variational inference for state space models](http://arxiv.org/abs/1511.07367)


## Contents
- Python Modules/Scripts
1. `run_model.py` Train tf_gbds model. Run `python run_model.py --help` to see [options](#train-an-tf_gbds-model).
2. `agents.py` The auxiliary class to construct computational graph (define generative and recognition models, draw samples, and implement one-step-ahead prediction).
3. `GenerativeModel.py` [The customized Edward Random Variables](http://edwardlib.org/api/model-development) which generate players' latent goal and control signal at each time point based on game state.
4. `RecognitionModel.py` The customized Edward Random Variables which infer the posterior goal and control signal using smoothing linear dynamical system. The code is based on [Evan Archer's implementation](https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py).
5. `utils.py` The utility functions needed for `run_model.py`.
6. `lib/` The directory containing library code for efficient matrix computation.

## How to Preprocess Your Data
Our model follows TensoFlow data input pipeline to read in experiment data as [TFRecord files](https://www.tensorflow.org/guide/datasets). Training and validation sets need to be saved separately. Only one field is required for each trial: `trajectory`, which our code expects to be a matrix with the following shape: (nTimepoints, nDimensions). Trial length can vary while the dimensionality must be consistent throughout the dataset.

- Optional fields that can be included
1. `extra_conds`: such as subect ID, type of opponent, or perhaps drug condition (i.e. saline v. muscimol). Our code expects extra conditions to be consistent within each trial.
2. `ctrl_obs`:  observed control signals with the same shape as `trajectory`.

## Prerequisites
The code is written in Python 3.6.x. You will also need:
* **TensorFlow** or **TensorFlow-GPU** version 1.6.0 ([install](https://www.tensorflow.org/install/)) (and [**TensorBoard**](https://www.tensorflow.org/guide/summaries_and_tensorboard) for [visualization](#visualize-a-training-model))
* **Edward** version 1.3.5 ([install](http://edwardlib.org/getting-started))
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)

## Before you start
[Clone this repository](https://help.github.com/articles/cloning-a-repository/) and run the following:
```sh
export PYTHONPATH=$PYTHONPATH:/path/to/your/directory/tf_gbds/
```
where "/path/to/your/directory" is replaced with the path to the tf_gbds repository. This allows the nested directories to access modules from their parent directory.

## Train an tf_gbds model
Once you prepare your dataset in the correct format, you can train the model! The following code runs the model with all of the default parameters (note that `model_dir`, `train_data_dir`, and `val_data_dir` are not set by default and must be provided by user):
```
# Run tf_gbds
$ python run_model.py --model_dir='new_model' (Directory where the model is saved) \
--train_data_dir='/directory/of/training/data' (Directory of training dataset file) \
--val_data_dir='/directory/of/validation/data' (Directory of validation dataset file) \
--synthetic_data=False (Is the model trained on synthetic data?) \
--save_posterior=True (Will posterior samples be retrieved after training?) \
--load_saved_model=False (Is the model restored from an existing checkpoint?)
--saved_model_dir='/directory/you/save/checkpoint' (Directory where the model to be restored is saved) \

--game_name="penaltykick" (Name of the game)
--n_agents=2 (Number of agents in the model)
--agent_name="goalie,shooter" (Name of each agent (separated by ,))
--agent_col="0;1,2" (Columns of dataset corresponding to each agent (separated by ; and ,))
--obs_dim=3 (Dimension of observation)
--extra_conds=Flase (Are extra conditions included in the dataset?)
--extra_dim=0 (Dimension of extra conditions)
--ctrl_obs=False (Are observed control signals included in the dataset?)
--add_accel=False (Is acceleration included in game state?")

--GMM_K=8 (Number of components in GMM) \
--gen_n_layers=3 (Number of layers in neural networks (generative model)) \
--gen_hidden_dim=64 (Number of hidden units in each dense layer of neural networks (generative model)) \
--rec_lag=10 (Number of previous timepoints included as input to recognition model) \
--rec_n_layers=3 (Number of layers in neural networks (recognition model)) \
--rec_hidden_dim=32 (Number of hidden units in each dense layer of neural networks (recognition model)) \

--sigma_init=-7. (Initial value of goal state variance) \
--sigma_trainable=False (Is sigma trainable?) \
--sigma_pen=1e3 (Penalty on large sigma) \
--g_lb=-1. (Goal state lower boundary) \
--g_ub=1. (Goal state upper boundary) \
--g_bounds_pen=None (Penalty for goal states escaping boundaries) \

--eps_init=-11. (Initial value of control signal variance) \
--eps_trainable=False (Is epsilon trainable?) \
--eps_pen=1e5 (Penalty on large epsilon) \
--latent_u=False (Is the true control signal modeled as latent variable?)
--clip=False (Is the observed control signal censored?) \
--clip_lb=-1. (Control signal censoring lower bound) \
--clip_ub=1. (Control signal censoring upper bound) \
--clip_tol=1e-5 (Tolerance of signal censoring) \
--clip_pen=1e8 (Penalty on control signal censoring)

--seed=1234 (Random seed) \
--opt='Adam' (Gradient descent optimizer) \
--lr=1e-3 (Initial learning rate) \
--n_epochs=500 (Number of iterations algorithm runs through the training set) \
--B=1 (Size of mini-batches) \
--n_samp=1 (Number of samples drawn for gradient estimation) \
--n_post_samp=30 (Number of samples from posterior distributions to draw and save) \
--max_ckpt=10 (Maximum number of checkpoints to keep in the directory) \
--freq_ckpt=5 (Frequency of saving checkpoints to the directory)
--freq_val_loss=1 (Frequency of computing validation set loss)
```

## Visualize a training model
To visualize training variables and loss curves while training tf_gbds model in **TensorBoard**, run the following command:
```sh
tensorboard --logdir=/directory/where/the/model/is/saved
```
