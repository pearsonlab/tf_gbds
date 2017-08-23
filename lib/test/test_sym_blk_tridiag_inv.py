import numpy as np
import tensorflow as tf
import numpy.testing as npt

import tf_gbds.lib.sym_blk_tridiag_inv as sym

# shared testing data
prec = np.float32
npA = np.array([[1, 6], [6, 4]]).astype(prec)
npB = np.array([[2, 7], [7, 4]]).astype(prec)
npC = np.array([[3, 9], [9, 1]]).astype(prec)
npD = np.array([[7, 2], [9, 3]]).astype(prec)

npZ = np.array([[0, 0], [0, 0]]).astype(prec)

npb = np.array([[0, 1], [2, 3], [4, 5], [6, 7]]).astype(prec)

fullmat = np.bmat([[npA,     npB, npZ,   npZ],
                   [npB.T,   npC, npD,   npZ],
                   [npZ,   npD.T, npC,   npB],
                   [npZ,     npZ, npB.T, npC]])


def test_compute_sym_blk_tridiag():

    alist = [fullmat[i:(i+2), i:(i+2)] for i in range(0, fullmat.shape[0], 2)]
    blist = [fullmat[(i+2):(i+4), i:(i+2)].T
             for i in range(0, fullmat.shape[0] - 2, 2)]

    AAi = tf.constant(np.array(alist))
    BBi = tf.constant(np.array(blist))

    fullmat_inv = np.linalg.inv(fullmat)

    D, OD, S = sym.compute_sym_blk_tridiag(AAi, BBi)

    with tf.Session() as sess:
        D_, OD_ = D.eval(), OD.eval()

    for (x, y) in zip(D_, [fullmat_inv[0:2, 0:2], fullmat_inv[2:4, 2:4],
                      fullmat_inv[4:6, 4:6], fullmat_inv[6:8, 6:8]]):
        npt.assert_allclose(x, y, atol=1e-4, rtol=1e-5)

    for (x, y) in zip(OD_, [fullmat_inv[2:4, 0:2], fullmat_inv[4:6, 2:4],
                      fullmat_inv[6:8, 4:6]]):
        npt.assert_allclose(x, y, atol=1e-4, rtol=1e-5)


def test_compute_sym_blk_tridiag_inv_b():

    alist = [fullmat[i:(i+2), i:(i+2)] for i in range(0, fullmat.shape[0], 2)]
    blist = ([fullmat[(i+2):(i+4), i:(i+2)].T
              for i in range(0, fullmat.shape[0] - 2, 2)])

    AAi = tf.constant(np.array(alist))
    BBi = tf.constant(np.array(blist))

    D, OD, S = sym.compute_sym_blk_tridiag(AAi, BBi)

    x = np.linalg.solve(fullmat, np.array([0, 1, 2, 3, 4, 5, 6, 7]))

    b = tf.expand_dims(tf.constant(npb), -1)

    tfx = sym.compute_sym_blk_tridiag_inv_b(S, D, b)

    with tf.Session() as sess:
        tfx_val = tfx.eval()
    npt.assert_allclose(tfx_val.flatten(), x, atol=1e-5, rtol=1e-4)
