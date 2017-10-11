'''
Run CGAN on GPU
Requires inputs to GAN (J (i.e. mu and lambda) and states) to be precomputed
for each trial and concatenated into one matrix for each. This is possible
since the CGAN is independent for each timepoint. Precomputing the inputs is
necessary because the other parts of the model (control model) aren't as easily
implemented for speedup on a GPU.
'''
import argparse
import tensorflow as tf
import sys
from os.path import join, expanduser
from utils import *
import numpy as np
import pickle as pkl
from tensorflow.contrib.keras import activations
from warnings import warn

# sys.path.append(expanduser('~/code/gbds/code/'))
# sys.path.append(expanduser('~/code/gbds/code/lib/'))


def train_model(**kwargs):
    directory = kwargs['directory']
    # if not (theano.config.device.startswith('gpu') or
    #         theano.config.device.startswith('cuda')):
    #     warn('GPU mode not activated in theano. Training on CPU...',
    #          RuntimeWarning)
    try:
        postJ = np.load(join(directory, 'postJ.npy'))
        states = np.load(join(directory, 'states.npy'))
        condition_dim = states.shape[1]
    except IOError:
        raise Exception('Inputs for CGAN not computed')

    # version = kwargs['version']
    batch_size = kwargs['batch_size']
    n_epochs = kwargs['n_epochs']
    n_extra_epochs = kwargs['n_extra_epochs']
    lmbda = kwargs['lambda']
    K = kwargs['K']
    init_G_iters = kwargs['init_G_iters']
    init_K = kwargs['init_K']
    lr = kwargs['lr']
    # compile_mode = kwargs['compile_mode']

    # load_GAN_model = kwargs['load_GAN_model']
    init_new_CGAN = kwargs['init_new_CGAN']
    nlayers_G = kwargs['nlayers_G']
    nlayers_D = kwargs['nlayers_D']
    noise_dim = kwargs['noise_dim']
    hidden_dim = kwargs['hidden_dim']
    condition_noise = kwargs['condition_noise']
    # nonlinearity = getattr(lasagne.nonlinearities, kwargs['nonlinearity'])
    leakiness = kwargs['leakiness']

    # instance_noise = (1.0, 0.0)  # bounds of instance noise over training
    instance_noise = None  # not in use currently

    np.random.seed(1234)
    data_iter_cgan = MultiDatasetMiniBatchIterator((postJ, states),
                                                   batch_size,
                                                   randomize=True)
    condition_scale = tf.reduce_max(states, axis=0, keep_dims=True)

    if instance_noise is not None:
        curr_inst_noise = tf.Variable(instance_noise[0], dtype=tf.float32)
        inst_noise_schedule = np.linspace(instance_noise[0],
                                          instance_noise[1],
                                          n_epochs, dtype=np.float32)
    else:
        curr_inst_noise = None

    if condition_noise is not None:
        curr_cond_noise = tf.Variable(condition_noise, dtype=tf.float32)
        cond_noise_schedule = np.linspace(condition_noise, 0.0,
                                          n_epochs, dtype=np.float32)
    else:
        curr_cond_noise = None

    # if load_GAN_model:
    #     with open(join(directory, 'model_gan%s.pkl' % version), 'rb') as f:
    #         model = pkl.load(f)
    #     try:
    #         model.mprior_goalie.CGAN_J.instance_noise = curr_inst_noise
    #         model.mprior_ball.CGAN_J.instance_noise = curr_inst_noise
    #     except AttributeError:
    #         print 'No CGAN_J initialized. Initializing new one.'
    #         init_new_CGAN = True
    #     try:
    #         cgan_g_cost = list(np.load(join(directory, 'cgan_g_costs%s.npy' % version)))
    #         cgan_d_cost = list(np.load(join(directory, 'cgan_d_costs%s.npy' % version)))
    #     except IOError:
    #         print 'No training cost function data available. Creating new data.'
    # else:
    #     with open(join(directory, 'model.pkl'), 'rb') as f:
    #         model = pkl.load(f)


    with tf.Session() as sess:
        print('Compiling graph for CGAN model...')
        sys.stdout.flush()
        # cgan_g_train_fn, cgan_d_train_fn = (
        #     compile_functions_cgan_training(model, lr,
        #                                     mode=compile_mode))
        saver = tf.train.import_meta_graph('gbds_test.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        if init_new_CGAN:
        model.mprior_ball.init_CGAN(condition_dim, noise_dim, hidden_dim,
                                    model.mprior_ball.JDim,
                                    nlayers_G=nlayers_G, nlayers_D=nlayers_D,
                                    batch_size=batch_size,
                                    lmbda=lmbda,
                                    condition_noise=curr_cond_noise,
                                    condition_scale=condition_scale,
                                    instance_noise=curr_inst_noise,
                                    nonlinearity=activations.relu(
                                        alpha=leakiness))
        model.mprior_goalie.init_CGAN(condition_dim, noise_dim, hidden_dim,
                                      model.mprior_goalie.JDim,
                                      nlayers_G=nlayers_G, nlayers_D=nlayers_D,
                                      batch_size=batch_size,
                                      lmbda=lmbda,
                                      condition_noise=curr_cond_noise,
                                      condition_scale=condition_scale,
                                      instance_noise=curr_inst_noise,
                                      nonlinearity=activations.relu(
                                          alpha=leakiness))
        cgan_g_cost = []
        cgan_d_cost = []

        print('---> Training GAN')
        sys.stdout.flush()

        if init_G_iters > 0:
            print('--> Early Training (Making D Optimal)')
            sys.stdout.flush()
        niter_tot = 0
        niter_g = 0
        init_done = False
        while not init_done:
            for curr_postJ, curr_states in data_iter_cgan:
                if niter_g == init_G_iters:
                    init_done = True
                    break
                if niter_tot % (init_K + 1) == init_K:
                    print('-> G iteration %i' % (niter_g + 1))
                    sys.stdout.flush()

                    model.set_training_mode('CGAN_G')
                    opt_g = tf.train.AdamOptimizer(lr)
                    train_g_op = opt_g.minimize(-model.cost(),
                                                var_list=model.getParams())
                    _, cost_g = sess.run([train_g_op, model.cost()],
                                         feed_dict={model.s: curr_states})
                    cgan_g_cost.append(cost_g)
                    for param in model.getParams():
                        if (param.name == 'W' or param.name == 'b' or
                                param.name == 'G'):
                            # only on NN parameters
                            param = tf.clip_by_norm(param, 5, axes=[0])
                    if np.isnan(cgan_g_cost[-1]):
                        raise Exception('NaN appeared')
                    niter_g += 1
                else:
                    model.set_training_mode('CGAN_D')
                    opt_d = tf.train.AdamOptimizer(lr)
                    train_d_op = opt_d.minimize(-model.cost(),
                                                var_list=model.getParams())
                    _, cost_d = sess.run([train_d_op, model.cost()],
                                         feed_dict={model.J: curr_postJ,
                                                    model.s: curr_states})
                    cgan_d_cost.append(cost_d)
                    if np.isnan(cgan_d_cost[-1]):
                        raise Exception('NaN appeared')
                niter_tot += 1

        # with open(directory + '/model_gan%s.pkl' % version, 'wb') as f:
        #     pkl.dump(model, f)
        # np.save(directory + '/cgan_g_costs%s' % version, cgan_g_cost)
        # np.save(directory + '/cgan_d_costs%s' % version, cgan_d_cost)
        saver.save(sess, directory + '/model_gan', write_meta_graph=True)
        np.save(directory + '/cgan_g_costs', cgan_g_cost)
        np.save(directory + '/cgan_d_costs', cgan_d_cost)

        print('--> Training GAN')
        sys.stdout.flush()
        niter = 0
        for ie in np.arange(n_epochs + n_extra_epochs):
            print('-> entering epoch %i' % (ie + 1))
            sys.stdout.flush()
            if instance_noise is not None and ie < n_epochs:
                curr_inst_noise.assign(inst_noise_schedule[ie])
            if condition_noise is not None and ie < n_epochs:
                curr_cond_noise.assign(cond_noise_schedule[ie])
            for curr_postJ, curr_states in data_iter_cgan:
                if niter % (K + 1) == K:
                    model.set_training_mode('CGAN_G')
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
                                         feed_dict={model.s: curr_states})
                    cgan_g_cost.append(cost_g)
                    if np.isnan(cgan_g_cost[-1]):
                        raise Exception('NaN appeared')
                else:
                    model.set_training_mode('CGAN_D')
                    opt_d = tf.train.AdamOptimizer(lr)
                    train_d_op = opt_d.minimize(-model.cost(),
                                                var_list=model.getParams())
                    _, cost_d = sess.run([train_d_op, model.cost()],
                                         feed_dict={model.J: curr_postJ,
                                                    model.s: curr_states})
                    cgan_d_cost.append(cost_d)
                    if np.isnan(cgan_d_cost[-1]):
                        raise Exception('NaN appeared')
                niter += 1
            # with open(directory + '/model_gan%s.pkl' % version, 'wb') as f:
            #     pkl.dump(model, f)
            # np.save(directory + '/cgan_g_costs%s' % version, cgan_g_cost)
            # np.save(directory + '/cgan_d_costs%s' % version, cgan_d_cost)
            if ie == 0:
                saver.save(sess, directory + '/model_gan', write_meta_graph=True)
            else:
                saver.save(sess, directory + '/model_gan', write_meta_graph=False)
            np.save(directory + '/cgan_g_costs', cgan_g_cost)
            np.save(directory + '/cgan_d_costs', cgan_d_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory of model')
    # parser.add_argument('--version', default='', type=str, help='Model version')
    parser.add_argument('--batch_size', default=5000, type=int, help='Batch size')
    parser.add_argument('--n_epochs', default=2000, type=int,
                        help='Number of training epochs')
    parser.add_argument('--n_extra_epochs', default=0, type=int,
                        help='Number of extra training epochs, after condition noise is 0')
    parser.add_argument('--lambda', default=10.0, type=float,
                        help='Penalty on gradient of discriminator')
    parser.add_argument('--K', default=3, type=int, help='D:G training ratio')
    parser.add_argument('--init_G_iters', default=25, type=int,
                        help='Number of Generator iterations for initialization')
    parser.add_argument('--init_K', default=100, type=int,
                        help='D:G ratio for initialization')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    # parser.add_argument('--compile_mode', default='FAST_RUN', type=str,
    #                     help='Theano compile mode')
    # parser.add_argument('--load_GAN_model', action='store_true',
    #                     help='Loads existing GAN model file (use if GAN_g0 is trained')
    parser.add_argument('--init_new_CGAN', action='store_true',
                        help='Initializes new CGAN')
    parser.add_argument('--nlayers_G', default=4, type=int,
                        help='Number of layers in Generator')
    parser.add_argument('--nlayers_D', default=4, type=int,
                        help='Number of layers in Discriminator')
    parser.add_argument('--noise_dim', default=15, type=int,
                        help='Number of noise dimensions')
    parser.add_argument('--hidden_dim', default=35, type=int,
                        help='Number of hidden units in G and D')
    parser.add_argument('--condition_noise', default=0.15, type=float,
                        help='Added noise to conditions')
    # parser.add_argument('--nonlinearity', default='leaky_rectify', type=str,
    #                     choices=['rectify', 'leaky_rectify', 'very_leaky_rectify'],
    #                     help='Nonlinearity type')
    parser.add_argument('--leakiness', default=0.01, type=float, # choices=[0, 0.01, 1/3],
                        help='Leakiness of rectify activation function')
    args = parser.parse_args()

    train_model(**vars(args))
