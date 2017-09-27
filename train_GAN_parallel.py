"""
Run GAN on GPU
Requires inputs to GAN (g0) to be precomputed
for each trial and concatenated into one matrix. Precomputing the inputs is
necessary because the other parts of the model (control model) aren't as easily
implemented for speedup on a GPU.
"""
import argparse
import tensorflow as tf
import sys
from os.path import join, expanduser
from utils import *
import numpy as np
import _pickle as pkl
from tensorflow.contrib.keras import activations
from warnings import warn

# sys.path.append(expanduser('~/code/gbds/code/'))
# sys.path.append(expanduser('~/code/gbds/code/lib/'))


def train_model(**kwargs):
    directory = kwargs['directory']
    # if not (theano.config.device.startswith('gpu') or
    #         theano.config.device.startswith('cuda')):
    #     warn("GPU mode not activated in theano. Training on CPU...",
    #          RuntimeWarning)
    try:
        post_g0 = np.load(join(directory, 'post_g0.npy'))
    except IOError:
        raise Exception("Inputs for GAN not computed")

    version = kwargs['version']
    batch_size = kwargs['batch_size']
    n_epochs = kwargs['n_epochs']
    n_extra_epochs = kwargs['n_extra_epochs']
    lmbda = kwargs['lambda']
    K = kwargs['K']
    init_G_iters = kwargs['init_G_iters']
    init_K = kwargs['init_K']
    lr = kwargs['lr']
    compile_mode = kwargs['compile_mode']

    load_GAN_model = kwargs['load_GAN_model']
    init_new_GAN = kwargs['init_new_GAN']
    nlayers_G = kwargs['nlayers_G']
    nlayers_D = kwargs['nlayers_D']
    noise_dim = kwargs['noise_dim']
    hidden_dim = kwargs['hidden_dim']
    nonlinearity = getattr(activations, kwargs['nonlinearity'])

    instance_noise = kwargs['instance_noise']

    np.random.seed(1234)
    data_iter_gan = DatasetMiniBatchIterator(post_g0,
                                             batch_size,
                                             randomize=True)

    if instance_noise is not None:
        curr_inst_noise = tf.Variable(instance_noise, dtype=tf.float32)
        inst_noise_schedule = np.linspace(instance_noise, 0.0,
                                          n_epochs, dtype=tf.float32)
    else:
        curr_inst_noise = None

    if load_GAN_model:
        with open(join(directory, 'model_gan%s.pkl' % version), 'rb') as f:
            model = pkl.load(f)
        try:
            model.mprior_goalie.GAN_g0.instance_noise = curr_inst_noise
            model.mprior_ball.GAN_g0.instance_noise = curr_inst_noise
        except AttributeError:
            print "No GAN_g0 initialized. Initializing new one."
            init_new_GAN = True
        try:
            gan_g_cost = list(np.load(join(
                directory, 'gan_g_costs%s.npy' % version)))
            gan_d_cost = list(np.load(join(
                directory, 'gan_d_costs%s.npy' % version)))
        except IOError:
            print "No training cost function data available. Creating new data."
    else:
        with open(join(directory, 'model.pkl'), 'rb') as f:
            model = pkl.load(f)

    if init_new_GAN:
        model.mprior_ball.init_GAN(noise_dim, hidden_dim,
                                   model.mprior_ball.yDim,
                                   nlayers_G=nlayers_G, nlayers_D=nlayers_D,
                                   batch_size=batch_size,
                                   lmbda=lmbda,
                                   instance_noise=curr_inst_noise,
                                   init_std_G=0.5,
                                   nonlinearity=nonlinearity)
        model.mprior_goalie.init_GAN(noise_dim, hidden_dim,
                                     model.mprior_goalie.yDim,
                                     nlayers_G=nlayers_G, nlayers_D=nlayers_D,
                                     batch_size=batch_size,
                                     lmbda=lmbda,
                                     instance_noise=curr_inst_noise,
                                     init_std_G=0.5,
                                     nonlinearity=nonlinearity)
        gan_g_cost = []
        gan_d_cost = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # print "Compiling graph for GAN model..."
        # sys.stdout.flush()
        # gan_g_train_fn, gan_d_train_fn = (
        #     compile_functions_gan_training(model, lr,
        #                                    mode=compile_mode))
        print '---> Training GAN'
        sys.stdout.flush()

        if init_G_iters > 0:
            print '--> Early Training (Making D Optimal)'
            sys.stdout.flush()
        niter_tot = 0
        niter_g = 0
        init_done = False
        while not init_done:
            for curr_post_g0 in data_iter_gan:
                if niter_g == init_G_iters:
                    init_done = True
                    break
                if niter_tot % (init_K + 1) == init_K:
                    print('-> G iteration %i' % (niter_g + 1))
                    sys.stdout.flush()

                    model.set_training_mode('GAN_G')
                    var_list = model.getParams()
                    for param in var_list:
                        if (param.name == 'W' or param.name == 'b' or
                                param.name == 'G'):
                        # only on NN parameters
                            param = tf.clip_by_norm(param, 5, axes=[0])
                    opt_g = tf.train.AdamOptimizer(lr)
                    train_g_op = opt_g.minimize(-model.cost(),
                                                var_list=var_list)
                    _, cost_g = sess.run([train_g_op, model.cost()],
                                         feed_dict={model.g0: curr_post_g0})
                    gan_g_cost.append(cost_g)
                    if np.isnan(gan_g_cost[-1]):
                        raise Exception("NaN appeared")
                    niter_g += 1
                else:
                    model.set_training_mode('GAN_D')
                    opt_d = tf.train.AdamOptimizer(lr)
                    train_d_op = opt_d.minimize(-model.cost(),
                                                var_list=model.getParams())
                    _, cost_d = sess.run([train_d_op, model.cost()],
                                         feed_dict={model.g0: curr_post_g0})
                    gan_d_cost.append(cost_d)
                    if np.isnan(gan_d_cost[-1]):
                        raise Exception("NaN appeared")
                niter_tot += 1

        with open(directory + '/model_gan%s.pkl' % version, 'wb') as f:
            pkl.dump(model, f)
        np.save(directory + '/gan_g_costs%s' % version, gan_g_cost)
        np.save(directory + '/gan_d_costs%s' % version, gan_d_cost)

        print '--> Training GAN'
        sys.stdout.flush()
        niter = 0
        for ie in np.arange(n_epochs + n_extra_epochs):
            if ie % 10 == 0:
                print('-> entering epoch %i' % (ie + 1))
                sys.stdout.flush()
            if instance_noise is not None and ie < n_epochs:
                curr_inst_noise.assign(inst_noise_schedule[ie])
            for curr_post_g0 in data_iter_gan:
                if niter % (K + 1) == K:
                    model.set_training_mode('GAN_G')
                    var_list = model.getParams()
                    for param in var_list:
                        if (param.name == 'W' or param.name == 'b' or
                                param.name == 'G'):
                        # only on NN parameters
                            param = tf.clip_by_norm(param, 5, axes=[0])
                    opt_g = tf.train.AdamOptimizer(lr)
                    train_g_op = opt_g.minimize(-model.cost(),
                                                var_list=var_list)
                    _, cost_g = sess.run([train_g_op, model.cost()],
                                         feed_dict={model.g0: curr_post_g0})
                    gan_g_cost.append(cost_g)
                    if np.isnan(gan_g_cost[-1]):
                        raise Exception("NaN appeared")
                else:
                    model.set_training_mode('GAN_D')
                    opt_d = tf.train.AdamOptimizer(lr)
                    train_d_op = opt_d.minimize(-model.cost(),
                                                var_list=model.getParams())
                    _, cost_d = sess.run([train_d_op, model.cost()],
                                         feed_dict={model.g0: curr_post_g0})
                    gan_d_cost.append(cost_d)
                    if np.isnan(gan_d_cost[-1]):
                        raise Exception("NaN appeared")

                niter += 1
            if ie % 10 == 0 or ie == n_epochs + n_extra_epochs - 1:
                with open(directory + '/model_gan%s.pkl' % version, 'wb') as f:
                    pkl.dump(model, f)
                np.save(directory + '/gan_g_costs%s' % version, gan_g_cost)
                np.save(directory + '/gan_d_costs%s' % version, gan_d_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Directory of model')
    parser.add_argument('--version', default='', type=str, help='Model version')
    parser.add_argument('--batch_size', default=500, type=int, help='Batch size')
    parser.add_argument('--n_epochs', default=6000, type=int,
                        help='Number of training epochs')
    parser.add_argument('--n_extra_epochs', default=3000, type=int,
                        help='Number of extra training epochs, after instance noise is 0')
    parser.add_argument('--lambda', default=10.0, type=float,
                        help='Penalty on gradient of discriminator')
    parser.add_argument('--K', default=1, type=int, help='D:G training ratio')
    parser.add_argument('--init_G_iters', default=50, type=int,
                        help='Number of Generator iterations for initialization')
    parser.add_argument('--init_K', default=100, type=int,
                        help='D:G ratio for initialization')
    parser.add_argument('--instance_noise', default=0.25, type=float,
                        help='Instance noise to add to input to D')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--compile_mode', default='FAST_RUN', type=str,
                        help='Theano compile mode')
    parser.add_argument('--load_GAN_model', action='store_true',
                        help='Loads existing GAN model file (use if CGAN_J is trained')
    parser.add_argument('--init_new_GAN', action='store_true',
                        help='Initializes new GAN')
    parser.add_argument('--nlayers_G', default=4, type=int,
                        help='Number of layers in Generator')
    parser.add_argument('--nlayers_D', default=4, type=int,
                        help='Number of layers in Discriminator')
    parser.add_argument('--noise_dim', default=5, type=int,
                        help='Number of noise dimensions')
    parser.add_argument('--hidden_dim', default=20, type=int,
                        help='Number of hidden units in G and D')
    parser.add_argument('--nonlinearity', default='leaky_rectify', type=str,
                        choices=['rectify', 'leaky_rectify', 'very_leaky_rectify'],
                        help='Nonlinearity type')

    args = parser.parse_args()
    train_model(**vars(args))
